// RADIANCE gfx1201 MXFP4 x FP8 GEMM: C[M,N] bf16 = A_fp8[M,K] . dequant(W_mxfp4)[N,K]^T
//
// Why a hand-written kernel: Triton lowers tl.dot_scaled by upconverting e2m1 to bf16 and using the
// 16-bit WMMA. Measured on this card, register-resident: fp8 WMMA 325.2 TFLOP/s vs f16 WMMA 160.2
// (2.03x), while Triton's own tl.dot manages 102.8 bf16 and only 43.3 on fp8 operands -- i.e. Triton
// will not emit the fp8 matrix instruction, it upconverts and pays conversion on top. The only way
// to the 2x is v_wmma_f32_16x16x16_fp8_fp8 by hand.
//
// Numerics: this is W4A8, not the checkpoint's declared W4A4. E2M1's 16 values are all exactly
// representable in e4m3 so the weight upconvert is a lossless table lookup, and the E8M0 scale is a
// power of two so applying it to the fp32 accumulator is exact. Activations are fp8 rather than fp4,
// which is strictly MORE precise than what the checkpoint was calibrated against -- but it does mean
// output is no longer bit-identical to the emulated path, so it is opt-in behind RADIANCE_MXFP4_W4A8.
//
// Layouts, matching what QuarkOCP_MX.process_weights_after_loading leaves on the layer:
//   A   [M,K]      e4m3, per-token scale As[M] applied in the epilogue
//   W   [N,K/2]    packed e2m1, low nibble = even k
//   Ws  [K/32,N]   e8m0, TRANSPOSED (the non-asm branch stores .T.contiguous()), which also makes
//                  the scale read coalesced across lanes
//
// Fragment layout confirmed empirically (0/256 elements wrong on a 16x16x16 tile):
//   A: lane l holds A[l%16][(l/16)*8+j] | B: B[(l/16)*8+j][l%16] | C: C[(l/16)*8+j][l%16]
//
// LDS rows are padded 8 B: without it, 16 lanes reading rows 64 B apart collide 8 ways on the
// 32 x 4 B banks. Staging through LDS is what makes this fast at all -- reading fragments straight
// from global ran at 9.5 TFLOP/s because lanes 16 rows apart touch 16 cache lines per operand.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>
#include <cstdlib>

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));

__device__ __constant__ unsigned char kLUT[16] = {
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC};

// kMag[d]: the 8 e2m1 magnitudes, each already shifted DOWN by d binades and encoded as an e4m3
// byte, packed 4 per word so `v_perm_b32` can resolve four lookups per instruction. Folding the MX
// block exponent into the WEIGHT here removes the per-32-block rescale from the inner loop
// entirely -- the scale becomes one per-row factor applied in the epilogue. The sign is separable
// (a negative code is the positive byte with bit 7 set), which is why only magnitudes are tabulated.
//
// The table runs into e4m3 SUBNORMALS, and that is deliberate. Stopping at the smallest e4m3 NORMAL
// (2^-6) makes the fold exact only while d <= 5, and this checkpoint exceeds that: mlp.down_proj
// reaches d=10 and holds 11,354 of the model's 14,433 out-of-range blocks. On layer 58 that zeroed
// 6.7% of the weights on 36 output channels, for per-channel errors up to 18.7% and a 1.18%
// whole-layer norm error -- squarely inside the range CHECKALL flags. Extending into subnormals
// (down to 2^-9) makes every d <= 8 exact and leaves d=9..12 rounding rather than flushing:
// measured, every affected layer with max d <= 8 goes to exactly zero error, and layer 58 (the only
// d=10 one) drops 4x to 0.31%.
//
// This is only sound because the gfx12 fp8 WMMA HONOURS e4m3 subnormals rather than flushing them
// -- verified on hardware, exact for every m*2^-9, m=1..7 (~/mxfp4_work/tier6/subnorm.hip).
__device__ __constant__ unsigned int kMag[16][2] = {
  {0x3c383000u, 0x4c484440u},
  {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u},
  {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u},
  {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u},
  {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u},
  {0x01010000u, 0x06040302u},
  {0x01000000u, 0x03020101u},
  {0x00000000u, 0x01010100u},
  {0x00000000u, 0x01000000u},
  {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u},
};

#define BM 128
#define BN 64
#define BK 64
#define PAD 8
#define ASTR (BK + PAD)
#define WSTR (BK + PAD)
#define NWAVE 8

__global__ __launch_bounds__(NWAVE * 32) void radiance_mxfp4_fp8_gemm(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const float *__restrict__ As,
    __bf16 *__restrict__ C, int M, int N, int K) {
  __shared__ unsigned char sA[BM * ASTR];
  __shared__ unsigned char sW[BN * WSTR];
  __shared__ unsigned char sS[BN * (BK / 32)];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave >> 1, wn = wave & 1;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int nblk = K / 32;

  floatx8 acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    {   // A tile: 32 B per thread. CLAMP, NEVER PREDICATE: a bounds-predicated staging load
        // lands in its own s_and_saveexec region and the compiler cannot carry a counted
        // s_wait_loadcnt across an exec-mask merge, so it emits s_wait_loadcnt 0x0 after every
        // load -- one full memory round trip each where one per slab would do. Clamped, all the
        // slab's loads issue back to back under a single wait. Safe because the epilogue already
        // drops rows past M and columns past N, and a clamped read returns real data. Measured on
        // the int4 twin (autoround-tests/RESULTS.md, 2026-08-26): worth 7-26% at decode and
        // 5-20% at prefill. Applies to every staging load below.
      const int r = tid >> 1, off = (tid & 1) * 32;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;
      const unsigned char *src = A + (size_t)(m0 + rc) * K + k0 + off;
      const uint4_t v0 = *(const uint4_t *)(src);
      const uint4_t v1 = *(const uint4_t *)(src + 16);
      *(uint4_t *)(&sA[r * ASTR + off]) = v0;
      *(uint4_t *)(&sA[r * ASTR + off + 16]) = v1;
    }
    {   // W tile: 8 packed bytes per thread -> 16 e4m3 bytes, guarded on the N tail
      const int r = tid >> 2, off = (tid & 3) * 8;
      const int rc = r < N - 1 - n0 ? r : N - 1 - n0;              // clamp, never predicate
      const unsigned long long packed =
          *(const unsigned long long *)(W + ((size_t)(n0 + rc) * K + k0) / 2 + off);
      unsigned char out[16];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const unsigned char byte = (packed >> (8 * t)) & 0xFF;
        out[2 * t] = kLUT[byte & 0xF];
        out[2 * t + 1] = kLUT[(byte >> 4) & 0xF];
      }
      *(uint4_t *)(&sW[r * WSTR + off * 2]) = *(const uint4_t *)out;
    }
    if (tid < BN * (BK / 32)) {   // scales are [K/32, N]: coalesced across n
      const int r = tid / (BK / 32), b = tid % (BK / 32);
      const int gn = n0 + r;
      const int gc = gn < N - 1 ? gn : N - 1;                      // clamp, never predicate
      sS[tid] = Ws[(size_t)(k0 / 32 + b) * N + gc];
    }
    __syncthreads();

#pragma unroll
    for (int blk = 0; blk < BK / 32; ++blk) {
      floatx8 tmp[2][2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) tmp[i][j][e] = 0.f;
#pragma unroll
      for (int step = 0; step < 2; ++step) {     // two WMMA steps span one 32-wide MX block
        const int kk = blk * 32 + step * 16 + kb8;
        int2_t af[2], wf[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const unsigned char *p = &sA[(wm * 32 + i * 16 + col) * ASTR + kk];
          af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const unsigned char *p = &sW[(wn * 32 + j * 16 + col) * WSTR + kk];
          wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            tmp[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], tmp[i][j]);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        // E8M0 is a power of two by definition, so 2^(e-127) is exactly the IEEE-754 float
        // whose exponent field is e: bits = e << 23. One integer shift instead of a
        // transcendental, worth +4% (exp2f ran 4x per lane per K-slab). e=0 would mean 2^-127
        // (denormal) and yields 0.0f here; it only arises for all-zero blocks, where the
        // product is 0 either way.
        const float s = __int_as_float(
            (int)sS[(wn * 32 + j * 16 + col) * (BK / 32) + blk] << 23);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][j][e] += tmp[i][j][e] * s;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * 32 + i * 16 + kb8 + e;
        const int n = n0 + wn * 32 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * As[m]);
      }
}


// Autotuned folded-scale kernel. The MX block exponent is pushed into the weight byte during the
// LDS upconvert (kMag), so the inner loop is pure WMMA -- no temp accumulator, no per-32-block
// rescale, no LDS scale reads. Lossless while the shifted value stays an e4m3 normal (d <= 5):
// 99.998% of the checkpoint's 761M blocks qualify, worst case 10, 282 blocks flush.
//
// Tile config from a correctness-gated sweep of 28 shapes (BM/BN/BK/TM/TN/wave grid). BM=256 via
// TM=4 won at EVERY production shape and batch size -- a taller tile buys more A reuse per weight
// read. It also removed the large-M cliff the earlier hand-written BM=128 version had.
#define TM 4
#define WM 4
#define WN 2
#define NWAVE (WM * WN)
#define NTHREADS (NWAVE * 32)
#define BMF (WM * TM * 16)
#define BNF_OF(tn) (WN * (tn) * 16)
#define BNF BNF_OF(2)

// TN is templated because the winner depends on M. Widening the N tile (TN=4 -> BNF=128) amortises
// the A-tile staging, which an ablation prices at 24% of runtime -- A is BMF*BK = 16 KB per K-slab
// against W's 2 KB, and with BNF=64 it is re-staged across twice as many column blocks. Measured on
// gate_up 17408x5120, tight A/B, 10 alternations with no overlap between the distributions:
//   M=8192 +10.0%  M=6144 +9.7%  M=4096 +8.5%  M=3072 +6.9%
//   M=2048 +1.3%   M=1536 -1.6%  M=1024 -4.2%  M=512 -8.8%
// Below the crossover the wide tile cannot fill: it halves the block count in N while BMF still
// masks most of a short M. Hence TN=4 only from M >= 2048; RADIANCE_MXFP4_TN4_MIN_M overrides.
// WPERM selects the weight layout. false is the checkpoint's own [N, K/2]; true is fragment
// order, one 32-lane uint32 slot per (n-tile, k-step), which is what libr4d's
// r4d_gemm_mxfp4a8_nt_m64 reads and what makes a wave's weight read 128 contiguous bytes instead
// of sixteen rows K/2 bytes apart. Both paths produce the SAME sW tile, so everything downstream
// of the staging loop is untouched.
template <int TN, bool WPERM, bool EPIFAST = true>
__global__ __launch_bounds__(NTHREADS) void radiance_mxfp4_fp8_gemm_folded(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  __shared__ unsigned char sA[BMF * ASTR];
  __shared__ unsigned char sW[WN * TN * 16 * WSTR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  constexpr int BNF_T = WN * TN * 16;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF_T;
  // Merged-linear partition select (paroquant: one launch over the whole N, partition p's n-blocks
  // read the p-th rotated copy of A and its scales). Boundaries are multiples of every n-tile
  // (launcher-checked), so a block never straddles one. pb1 = pb2 = 1<<30 (the default) is p = 0.
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    A += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }

  floatx8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Wave-uniform 64-bit bases, hoisted out of the per-thread address arithmetic. Written as
    // `base + int_offset`, the compiler keeps the base in SGPRs and emits the SADDR form of
    // global_load, so a lane pays one 32-bit add instead of the v_add_co_u32 / v_add_co_ci_u32
    // pair a full 64-bit address costs. The matmul block was spending ~71 of its 565 instructions
    // on that arithmetic while using only 29 of the ~102 available SGPRs.
    // r*K fits an int by a wide margin: r < BMF = 256 and K <= 17408, so r*K < 4.5e6.
    const unsigned char *__restrict__ Ab = A + (size_t)m0 * K + k0;
    const unsigned char *__restrict__ Wb = W + ((size_t)n0 * K + k0) / 2;
    // s_prefetch_data on the next slab's weight runs: measured and REJECTED 2026-08-28 --
    // -0.3 to -1% on every production shape (tier7 spfd mode). With branchless staging at
    // occupancy 10 the loads are already fully overlapped; early L2 arrival buys nothing.
    const unsigned char *__restrict__ Wsb = Ws + (size_t)(k0 / 32) * N + n0;
    const unsigned char *__restrict__ Wrefb = Wref + n0;
    // ---- A tile: BM*BK bytes, 16 B per thread per iteration ----
#pragma unroll
    for (int off = 0; off < BMF * BK; off += NTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / BK, c = idx % BK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;              // clamp, never predicate
      *(uint4_t *)(&sA[r * ASTR + c]) = *(const uint4_t *)(Ab + (rc * K + c));
    }
    // ---- W tile: BN*BK/2 packed bytes, 8 B per thread per iteration -> 16 e4m3 bytes ----
    if constexpr (WPERM) {
      // Fragment order: slot s of the tile is one lane's four packed bytes (eight elements) for
      // one (n-tile, k-step), so consecutive threads read consecutive uint32 and a wave takes 128
      // contiguous bytes. WSLOTS adjacent lane slots per thread; adjacent slots differ only in
      // the row (lane index is half*16 + row), so one vector read covers WSLOTS adjacent rows at
      // the same k and the wave still reads its bytes contiguously.
      //
      // WSLOTS is 4 (a 16 B request) whenever the tile has the slots for it, else 2 (8 B). The
      // original 2-slot trade was priced against PREDICATED staging: one slot (4 B) cost 14-25%
      // of prefill, two recovered most of it, and at TN=4 the remaining narrow-request penalty
      // was ~11% against checkpoint order (tier7, 2026-08-28). A 4-slot group never straddles
      // the k-half boundary because 16 % 4 == 0.
      constexpr int KSTEPS_T = BK / 16, NTILES_T = BNF_T / 16;
      constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
      constexpr int WSLOTS = (TOT_SLOTS >= NTHREADS * 4) ? 4 : 2;
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < TOT_SLOTS; off += NTHREADS * WSLOTS) {
        const int sl = off + tid * WSLOTS;              // first of the group; WSLOTS-aligned
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
        // Clamp, never predicate (see the A-tile note in the first kernel). gn is always
        // WSLOTS-aligned here and N % 16 == 0 under WPERM, so clamping to the last WSLOTS-aligned
        // row keeps the vector read aligned and every gc + q in bounds; the fragment address is
        // recomputed from the clamped row.
        const int gc = gn < N - WSLOTS ? gn : N - WSLOTS;
        const int lanec = (lane_ & 16) | (gc & 15);
        const unsigned int *src = &Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec];
        unsigned int wq[WSLOTS];
        if constexpr (WSLOTS == 4) {
          const uint4_t v = *(const uint4_t *)src;
          wq[0] = v[0]; wq[1] = v[1]; wq[2] = v[2]; wq[3] = v[3];
        } else {
          const uint2_t v = *(const uint2_t *)src;
          wq[0] = v[0]; wq[1] = v[1];
        }
        const int blk = (k0 + kst * 16) / 32;
        // One dword read each for the WSLOTS reference and block exponents instead of 2*WSLOTS
        // byte reads: gc is WSLOTS-aligned and N % 16 == 0, so both reads are aligned.
        unsigned int refw, wsw;
        if constexpr (WSLOTS == 4) {
          refw = *(const unsigned int *)(Wref + gc);
          wsw = *(const unsigned int *)(Ws + (size_t)blk * N + gc);
        } else {
          refw = *(const unsigned short *)(Wref + gc);
          wsw = *(const unsigned short *)(Ws + (size_t)blk * N + gc);
        }
#pragma unroll
        for (int q = 0; q < WSLOTS; ++q) {
          int d = (int)((refw >> (8 * q)) & 0xFF) - (int)((wsw >> (8 * q)) & 0xFF);
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
          const unsigned int ev = wq[q] & 0x0F0F0F0Fu;
          const unsigned int od = (wq[q] >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(uint2_t *)(&sW[(r + q) * WSTR + kloc]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    } else
#pragma unroll
    for (int off = 0; off < BNF_T * BK / 2; off += NTHREADS * 8) {
      const int idx = off + tid * 8;
      const int r = idx / (BK / 2), c = idx % (BK / 2);
      const int rc = r < N - 1 - n0 ? r : N - 1 - n0;          // clamp, never predicate
      // Same hoist as A. rc*(K/2) < 1.2e6 and the scale offset < 3*N, both comfortably int.
      const unsigned long long packed = *(const unsigned long long *)(Wb + (rc * (K / 2) + c));
      const int blk = (c * 2) / 32;                            // k0's share is in Wsb
      int dsh = (int)Wrefb[rc] - (int)Wsb[blk * N + rc];       // scales are [K/32, N]
      dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
      // v_perm_b32 resolves 4 table lookups per instruction from the 8-byte magnitude table;
      // the sign is separable because entries 8-15 are entries 0-7 with bit 7 set. The upconvert
      // measured 21% of runtime with per-byte constant loads (156 -> 188 TF with it removed
      // entirely), and this recovers +9-12% of that.
      const unsigned int t0 = kMag[dsh][0], t1 = kMag[dsh][1];
      unsigned int outw[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int w = (unsigned int)(packed >> (32 * h));
        const unsigned int ev = w & 0x0F0F0F0Fu;
        const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
        outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
      }
      *(uint4_t *)(&sW[r * WSTR + c * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
    }
    __syncthreads();

#pragma unroll
    for (int step = 0; step < BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[TM], wf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const unsigned char *p = &sA[(wm * TM * 16 + i * 16 + col) * ASTR + kk];
        af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * WSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      // Hard scheduling barrier per k-step. Without it the compiler hoists EVERY step's fragment
      // loads above the first WMMA -- the ISA shows 346 instructions and 140 v_dual_mov_b32 ahead
      // of a 64-WMMA run that itself contains zero moves. Those moves are the register shuffling
      // that holds four k-steps of fragments live at once. Fencing each step keeps one step's
      // fragments live instead of four. Same idea as the attention kernel's SGB bit, which notes
      // that without the barrier its prefetch is inert.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }

  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
  // Epilogue addressing and predication. `C[(size_t)m * N + n]` is a full 64-bit address per
  // element and there are TM*TN*8 of them, each in its own s_and_saveexec region from the bounds
  // test. Measured in the ISA: 145 v_add_co_u32/v_add_co_ci pairs, 103 s_and_saveexec and 130
  // 64-bit shifts in a kernel with 32 WMMA -- about 15% of the instruction stream spent getting
  // to memory rather than computing.
  //
  // Hoisting a wave-uniform base and indexing with a 32-bit offset gets the SADDR form, so a lane
  // pays one 32-bit add instead of a 64-bit add pair -- the same reasoning as the staging bases
  // above. And a block lying entirely inside M and N needs no bounds test at all; on a real
  // prefill only the last row-block and column-block are ragged. The offset cannot overflow an
  // int: at most (kb8 + (TM-1)*16 + 7) * N + N <= 64 * 34816.
  //
  // BIT-IDENTICAL to the path below by construction -- same operands, same order, same rounding.
  // It is gated that way rather than on a tolerance, because nothing here may change numerics.
  if constexpr (EPIFAST) {
    __bf16 *__restrict__ Cb = C + (size_t)(m0 + wm * TM * 16) * N;
    const float *__restrict__ Asb = As + m0 + wm * TM * 16;
    const bool full = (m0 + wm * TM * 16 + (TM - 1) * 16 + kb8 + 7 < M) &&
                      (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
    if (full) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = i * 16 + kb8 + e;
            const int n = n0 + wn * TN * 16 + j * 16 + col;
            Cb[r * N + n] = (__bf16)(acc[i][j][e] * rf[j] * Asb[r]);
          }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * rf[j] * As[m]);
      }
}


// ---------------------------------------------------------------- A-tiled prefill kernel
// Same tile, same W staging, same epilogue as the folded kernel above, but the activation
// arrives in WMMA-FRAGMENT-TILED layout: each 16m x 16k fp8 fragment is 256 contiguous bytes in
// lane order (lane l owns bytes 8l..8l+7 = A[mt*16 + l%16][ks*16 + (l/16)*8 .. +7]), the layout
// radiance_add_rms_quant / radiance_silu_mul_quant emit under RADIANCE_MXFP4_A_TILED_MIN_M. One
// coalesced global_load_b64 per fragment per wave lands straight in the af[] register the WMMA
// reads, so sA and its staging (the largest single cost of the folded kernel: ablated at 24-32%)
// are gone; only W goes through LDS (9-18 KB), so 3 blocks fit a CU instead of 2.
//
// Measured 2026-09-01 (~/mxfp4_work/tier7/lf.hip, DRAM-fed, prod server up, bit-identical on all
// six shapes x M tails x TN): vs the shipped folded kernel on the checkpoint layout, 0.84-0.87x
// at M=2048 (TN=4), 0.87-0.91x at M=1024 (TN=2), 0.878x at out M=8192 and 0.868x at down M=4096
// -- 215-220 TF/s against 190. The fragment-order (WPERM) layout is within 1% of that, so this
// kernel also removes the old "wp1 costs prefill" trade. Rejected on the way, all measured:
// W direct-to-register with an in-register fold (no LDS at all) 1.10-1.30x SLOWER at every
// shape -- the fold is then done by all WM=4 waves and the 4 B/lane W loads add instructions;
// a register-pipelined A prefetch (next slab in flight) 1.05-1.10x slower than this plain
// form at 250 VGPRs, and it spills at TN=4 with a 64-k slab (93 VGPRs, 2.2-4.9x).
// LBK: K per slab. With no A tile in LDS a 128-deep W slab is affordable (17 KB at TN=4, 3
// blocks/CU) and halves the barriers per K. Measured 2026-09-02 (tier7 lf.hip, bit-identical):
// at TN=2 (M < 2048) LBK=128 is 2-4% faster than 64 on every shape (gate48 -22%); at TN=4 it
// costs registers (246 VGPRs, occupancy 5) and loses 12-17%, so the launcher keys LBK on TN.
//
// ROUND 2 (2026-09-03, ~/mxfp4_work/tier7/at2.{h,hip}, abl.hip). An ablation of this kernel
// (WMMA count held, one piece removed at a time, TN=4 M=4096) priced its non-WMMA time: A
// fragment loads 28% (of which ~18% is the L2->L0 stream and ~10% issue/latency), W staging
// 14% (8% of it the fold VALU), LDS fragment reads 11%, barriers 1%; WMMA-only runs at 310
// TF/s = 55% of the full time, and WMMA-only + everything-else sum to ~90%, i.e. the memory
// work barely overlaps the matrix work. Three codegen fixes, each bit-identical, together
// 0.94-0.96x paired vs the 09-02 kernel on every shape at BOTH bands and both W layouts:
//   * SGPR bases for the A fragments (readfirstlane on the tile offset) with an UNSIGNED 32-bit
//     lane offset, so the 16 loads compile to one s_clause of SADDR global_load_b64 with
//     immediate offsets instead of eight 64-bit v_add_co/v_add_co_ci chains and their
//     s_wait_alu stalls (main loop VALU 63 -> ~47 per 64 WMMA).
//   * kMag served from LDS. As a global (constant) load it is a DEPENDENT load issued after the
//     A fragments, and loadcnt is in order, so `s_wait_loadcnt 0x0` before the fold drained the
//     whole A batch every slab. A ds_load has its own counter.
//   * LDS-only barrier fences. __syncthreads() is a workgroup-scope acq/rel on ALL address
//     spaces: `global_inv` plus a full loadcnt drain, twice per slab. Nothing in global memory
//     is shared between waves here, so the fence is scoped to LDS (s_wait_dscnt + s_barrier).
// Priced and REJECTED in the same round (all gated bit-identical): A fragments software-
// pipelined one slab ahead (loop unrolled by two, 230 VGPRs at TN=4): neutral on top of the
// three fixes at TN=4 and -2..-5% at TN=2 from the occupancy loss (the A cost is the L2 stream,
// not latency -- a cache-hot A ablation is still 10% off no-A); prefetching next slab's W into
// registers: +-1%. Every rebalancing of the tile trades one memory stream for another (TN=8 is
// the accumulator wall; A shared via LDS is the folded kernel; TM=3/TN=6 nets ~0), so the
// remaining ~40% over the WMMA floor is structural at this tile.
// Served 2026-09-04 (serve-mxfp4.sh defaults, BetterBench quick --prefill, same-day paired vs
// the 09-02 kernel): PP t/s 5141->5274 @2k, 5283->5385 @8k, 5109->5203 @16k, 4937->5046 @32k,
// 4686->4794 @64k (+1.8..+2.6%); decode untouched (this kernel only runs at M >= 513).
static __device__ __forceinline__ void radiance_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <int TN, bool WPERM, int LBK = BK>
__global__ __launch_bounds__(NTHREADS) void radiance_mxfp4_fp8_gemm_atiled(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, __bf16 *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  constexpr int NS = LBK / 16;
  constexpr int LWSTR = LBK + PAD;
  constexpr int BNF_T = WN * TN * 16;
  constexpr int NIT = BNF_T * LBK / 2 / (NTHREADS * 8);   // row-major W staging iterations per slab
  static_assert(NIT >= 1 && NIT * NTHREADS * 8 == BNF_T * LBK / 2, "W tile must be 8 B/thread multiples");
  __shared__ unsigned char sW[BNF_T * LWSTR];
  __shared__ unsigned int sMag[32];   // kMag, 128 B: ds_load keeps the lookup off the loadcnt chain

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF_T;
  // Merged-linear partition select (paroquant: one launch over the whole N, partition p's n-blocks
  // read the p-th rotated copy of A and its scales). Boundaries are multiples of every n-tile
  // (launcher-checked), so a block never straddles one. pb1 = pb2 = 1<<30 (the default) is p = 0.
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    AT += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }
  const int ksteps_g = K / 16;
  const int Mt = (M + 15) >> 4;

  if (tid < 32) sMag[tid] = ((const unsigned int *)kMag)[tid];
  radiance_lds_barrier();

  // A tile bases, one per M-fragment: wave-uniform, forced into SGPRs; the lane part is an
  // unsigned 32-bit offset so the loads take the SADDR form (max offset < 2^31 for all prod).
  // Tile-granular clamp, never predicate: rows of a clamped tile only reach accumulators the
  // epilogue drops.
  const unsigned char *abase[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int mt = (m0 >> 4) + wm * TM + i; mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps_g * 256);
  }
  const unsigned int aoff = lane * 8;

  // W staging geometry, hoisted out of the K loop.
  //   Row-major (checkpoint) layout: row r, packed-byte column c, clamped row rc; 8 B/thread.
  //   Fragment order (WPERM): slot s of the tile is one lane's four packed bytes (eight
  //   elements) for one (n-tile, k-step), so consecutive threads read consecutive uint32 and a
  //   wave takes 128 contiguous bytes. WSLOTS adjacent lane slots per thread; adjacent slots
  //   differ only in the row (lane index is half*16 + row), so one vector read covers WSLOTS
  //   adjacent rows at the same k and the wave still reads its bytes contiguously. WSLOTS is 4
  //   (16 B) whenever the tile has the slots for it, else 2 (8 B); a 4-slot group never
  //   straddles the k-half boundary because 16 % 4 == 0. Clamp to the last WSLOTS-aligned row
  //   (N % 16 == 0 under WPERM) so the vector read stays aligned and every gc + q is in bounds.
  constexpr int KSTEPS_T = LBK / 16, NTILES_T = BNF_T / 16;
  constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
  constexpr int WSLOTS = (TOT_SLOTS >= NTHREADS * 4) ? 4 : 2;
  constexpr int NITW = TOT_SLOTS / (NTHREADS * WSLOTS);
  static_assert(!WPERM || NITW * NTHREADS * WSLOTS == TOT_SLOTS, "WPERM slot groups must tile");
  constexpr int NW = WPERM ? NITW : NIT;
  int wrow[NW], wcol[NW], wrc[NW], wkst[NW], wlanec[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    if constexpr (WPERM) {
      const int sl = it * NTHREADS * WSLOTS + tid * WSLOTS;
      const int lane_ = sl & 31, rest = sl >> 5;
      const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
      wrow[it] = ntl * 16 + (lane_ & 15);
      wcol[it] = kst * 16 + (lane_ >> 4) * 8;          // kloc
      const int gn = n0 + wrow[it];
      wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;      // global clamped row
      wlanec[it] = (lane_ & 16) | (wrc[it] & 15);
      wkst[it] = kst;
    } else {
      const int idx = it * NTHREADS * 8 + tid * 8;
      wrow[it] = idx / (LBK / 2); wcol[it] = idx % (LBK / 2);
      wrc[it] = wrow[it] < N - 1 - n0 ? wrow[it] : N - 1 - n0;
      wkst[it] = 0; wlanec[it] = 0;
    }
  }
  unsigned int wref[NW];   // per-row max exponent(s): one byte, or WSLOTS packed under WPERM
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    if constexpr (WPERM) {
      if constexpr (WSLOTS == 4) wref[it] = *(const unsigned int *)(Wref + wrc[it]);
      else                       wref[it] = *(const unsigned short *)(Wref + wrc[it]);
    } else wref[it] = Wref[n0 + wrc[it]];
  }

  floatx8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += LBK) {
    // A fragments straight from global: one SADDR clause, consumed only at the WMMAs.
    int2_t af[TM][NS];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const int2_t *)(abase[i] + (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));

    // ---- W tile -> LDS, folded to e4m3 (kMag[d] via v_perm, see the folded kernel) ----
    uint4_t pk[NW]; unsigned int wsv[NW];
    if constexpr (WPERM) {
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int kstep0 = k0 / 16;
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        const unsigned int *src = &Wp[((size_t)(wrc[it] >> 4) * ksteps_g + kstep0 + wkst[it]) * 32 + wlanec[it]];
        if constexpr (WSLOTS == 4) pk[it] = *(const uint4_t *)src;
        else { const uint2_t v = *(const uint2_t *)src; pk[it] = uint4_t{v[0], v[1], 0u, 0u}; }
        const int blk = (k0 + wkst[it] * 16) / 32;
        if constexpr (WSLOTS == 4) wsv[it] = *(const unsigned int *)(Ws + (size_t)blk * N + wrc[it]);
        else                       wsv[it] = *(const unsigned short *)(Ws + (size_t)blk * N + wrc[it]);
      }
    } else {
      const unsigned char *__restrict__ Wb = W + (size_t)__builtin_amdgcn_readfirstlane((int)(((size_t)n0 * K + k0) / 2));
      const unsigned char *__restrict__ Wsb = Ws + (size_t)__builtin_amdgcn_readfirstlane((k0 / 32) * N + n0);
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        const uint2_t v = *(const uint2_t *)(Wb + (unsigned int)(wrc[it] * (K / 2) + wcol[it]));
        pk[it] = uint4_t{v[0], v[1], 0u, 0u};
        wsv[it] = Wsb[(unsigned int)(((wcol[it] * 2) / 32) * N + wrc[it])];
      }
    }
    if constexpr (WPERM) {
#pragma unroll
      for (int it = 0; it < NW; ++it)
#pragma unroll
        for (int q = 0; q < WSLOTS; ++q) {
          int d = (int)((wref[it] >> (8 * q)) & 0xFF) - (int)((wsv[it] >> (8 * q)) & 0xFF);
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const uint2_t t = *(const uint2_t *)&sMag[d * 2];
          const unsigned int wq = pk[it][q];
          const unsigned int ev = wq & 0x0F0F0F0Fu;
          const unsigned int od = (wq >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t[1], t[0], ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t[1], t[0], od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(uint2_t *)(&sW[(wrow[it] + q) * LWSTR + wcol[it]]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
    } else {
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        int dsh = (int)wref[it] - (int)wsv[it];
        dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
        const uint2_t t = *(const uint2_t *)&sMag[dsh * 2];
        unsigned int outw[4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const unsigned int w = pk[it][h];
          const unsigned int ev = w & 0x0F0F0F0Fu;
          const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t[1], t[0], ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t[1], t[0], od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
          outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
        }
        *(uint4_t *)(&sW[wrow[it] * LWSTR + wcol[it] * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
      }
    }
    radiance_lds_barrier();

#pragma unroll
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      int2_t wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * LWSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][step], wf[j], acc[i][j]);
    }
    radiance_lds_barrier();
  }

  // Epilogue: identical to the folded kernel's EPIFAST path (bit-identical by construction).
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
  {
    __bf16 *__restrict__ Cb = C + (size_t)(m0 + wm * TM * 16) * N;
    const float *__restrict__ Asb = As + m0 + wm * TM * 16;
    const bool full = (m0 + wm * TM * 16 + (TM - 1) * 16 + kb8 + 7 < M) &&
                      (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
    if (full) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = i * 16 + kb8 + e;
            const int n = n0 + wn * TN * 16 + j * 16 + col;
            Cb[r * N + n] = (__bf16)(acc[i][j][e] * rf[j] * Asb[r]);
          }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)(acc[i][j][e] * rf[j] * As[m]);
      }
}

// ---------------------------------------------------------------- decode path (M <= 16)
// The tiles above are sized for prefill: BMF=256 via TM=4. At decode M is 5 (batch 1 x SPEC+1), so
// per wave the folded kernel issues K/DBK * DBK/16 * TM * TN = 4352 WMMA against 5 real rows -- 51x
// more MACs than useful, a ~70 us matrix-issue floor on a shape whose DRAM floor is 41 us. TM=1
// cuts issued MACs 16x and shrinks the A tile from 256xBK to 16xBK at the same time.
//
// Split-K is not optional here: with TM=1 the independent N-fragments number N/16, so N=5120 gives
// only 320 waves = 2.5 per SIMD no matter how the N tile is chosen. Measured, DKS=1 is 2.4x worse
// than DKS=4 on mlp.down. Partials are fp32 [DKS][M][N]; the folded scale composes for free because
// Wref[n] is a whole-row max over k, so both per-row factors are applied once in the reduce.
//
// DBK=128 here, which REVERSES the prefill answer (where it measured -34%). That loss was pure LDS
// occupancy: 22.5 -> 43.5 KB took resident blocks 2 -> 1. With a 16-row A tile, LDS is
// (16+BND)*(DBK+PAD) = 10.88 KB, so 5-6 blocks still fit. Measured 1.87x on gate_up. See
// ~/mxfp4_work/tier5/RESULTS.md.
#define DEC_MTILE 16
#define DEC_PAD 8
static __device__ __forceinline__ unsigned int wq_dec(uint2_t v, int q) { return q ? v[1] : v[0]; }
// Nontemporal (streaming) weight loads, `global_load ... th:TH_LOAD_NT` on gfx1201. The decode
// weight stream is read exactly once per step, so a streaming hint keeps it from displacing the
// split-K partials and A in L2/MALL. Applied to the W loads ONLY: Ws/Wref are re-read per slab and
// the partials must stay cached for the fused reduce. RADIANCE_MXFP4_DECODE_NT=1 selects it.
template <bool NT, typename T>
static __device__ __forceinline__ T ld_w(const T *p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

// DTM = number of 16-row M-fragments; the launcher picks ceil(M/16). vLLM captures decode at
// num_seqs*(SPEC+1): 5/10/20/25/35/40 under mtp at SPEC=4, and batch x 8 under a dflash drafter
// at SPEC=7, which reaches 64 at max_num_seqs=8. DTM 1..4 covers every reachable batch of both.
// Use the SMALLEST DTM that covers M: measured at M=20, DTM=2 beats DTM=3 by 12% (89.1 vs 99.9 us
// on gate_up) purely because the wider tile computes rows nobody asked for.
template <int DWN, int DBK, int DKS, int DTM, bool WPERM, bool DIRECT = true, bool NT = false>
__global__ __launch_bounds__(DWN * 32) void radiance_mxfp4_fp8_gemm_decode(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    __bf16 *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride) {
  constexpr int BND = DWN * 16;
  constexpr int DASTR = DBK + DEC_PAD, DWSTR = DBK + DEC_PAD;
  constexpr int DNTHREADS = DWN * 32;
  __shared__ unsigned char sA[DEC_MTILE * DTM * DASTR];
  __shared__ unsigned char sW[BND * DWSTR];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;
  // Merged-linear partition select (paroquant: one launch over the whole N, partition p's n-blocks
  // read the p-th rotated copy of A and its scales). Boundaries are multiples of every n-tile
  // (launcher-checked), so a block never straddles one. pb1 = pb2 = 1<<30 (the default) is p = 0.
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    A += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }

  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);

  floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  // ---- per-slab pieces, shared by the three slab loops below ----
  auto stage_a = [&](int k0) {
#pragma unroll
    for (int off = 0; off < DEC_MTILE * DTM * DBK; off += DNTHREADS * 16) {
      const int idx = off + tid * 16;
      if (idx < DEC_MTILE * DTM * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;                  // clamp, never predicate
        *(uint4_t *)(&sA[r * DASTR + c]) = *(const uint4_t *)(A + (size_t)rc * K + k0 + c);
      }
    }
  };
  // Row-order (checkpoint layout) W staging from a packed 8-byte word per (iteration, thread).
  // getw(it, gc, c) supplies the word so the plain and line-group loops share the fold.
  constexpr int NITW = BND * DBK / 2 / (DNTHREADS * 8);
  auto stage_w_row = [&](int k0, auto &&getw) {
#pragma unroll
    for (int it = 0; it < NITW; ++it) {
      const int idx = it * DNTHREADS * 8 + tid * 8;
      const int r = idx / (DBK / 2), c = idx % (DBK / 2), gn = n0 + r;
      const int gc = gn < N - 1 ? gn : N - 1;                  // clamp, never predicate
      const unsigned long long packed = getw(it, gc, c);
      const int blk = (k0 + c * 2) / 32;
      int dsh = (int)Wref[gc] - (int)Ws[(size_t)blk * N + gc];
      dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
      const unsigned int t0 = kMag[dsh][0], t1 = kMag[dsh][1];
      unsigned int outw[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int wv = (unsigned int)(packed >> (32 * h));
        const unsigned int ev = wv & 0x0F0F0F0Fu;
        const unsigned int od = (wv >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        outw[2 * h]     = __builtin_amdgcn_perm(bo, be, 0x05010400u);
        outw[2 * h + 1] = __builtin_amdgcn_perm(bo, be, 0x07030602u);
      }
      *(uint4_t *)(&sW[r * DWSTR + c * 2]) = uint4_t{outw[0], outw[1], outw[2], outw[3]};
    }
  };
  auto stage_w_wperm = [&](int k0) {
    if constexpr (WPERM) {
      // Fragment order, as in the folded kernel: consecutive threads take consecutive uint32, so
      // a wave's weight read is 128 contiguous bytes instead of sixteen rows K/2 bytes apart.
      // TWO lane slots per thread (an 8 B request), the width the int4 twin measured optimal
      // (4 B was 11% slower there; 16 B identical). Adjacent slots are adjacent rows at the same
      // k, exactly as in the folded kernel above.
      constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < NTILES_T * KSTEPS_T * 32; off += DNTHREADS * 2) {
        const int sl = off + tid * 2;                   // first of the pair; always even
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
        // Clamp, never predicate; gn is always even here and N % 16 == 0 under WPERM, so
        // clamping to the last even row keeps the uint2 read 8-byte aligned and gc + 1 in
        // bounds; the fragment address is recomputed from the clamped row.
        const int gc = gn < N - 2 ? gn : N - 2;
        const int lanec = (lane_ & 16) | (gc & 15);
        const uint2_t wv = ld_w<NT>(
            (const uint2_t *)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec]);
        const int blk = (k0 + kst * 16) / 32;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          int d = (int)Wref[gc + q] - (int)Ws[(size_t)blk * N + gc + q];
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
          const unsigned int ev = wq_dec(wv, q) & 0x0F0F0F0Fu;
          const unsigned int od = (wq_dec(wv, q) >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(uint2_t *)(&sW[(r + q) * DWSTR + kloc]) =
              uint2_t{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    }
  };
  auto mma_slab = [&]() {
    __syncthreads();
#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[DTM], wf;
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char *pa = &sA[(i * 16 + col) * DASTR + kk];
        af[i][0] = *(const int *)pa; af[i][1] = *(const int *)(pa + 4);
      }
      const unsigned char *pw = &sW[(wave * 16 + col) * DWSTR + kk];
      wf[0] = *(const int *)pw; wf[1] = *(const int *)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[i]);
    }
    __syncthreads();
  };

  if constexpr (WPERM) {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_wperm(k0);
      mma_slab();
    }
  } else {
    for (int s = s_lo; s < s_hi; ++s) {
      const int k0 = s * DBK;
      stage_a(k0);
      stage_w_row(k0, [&](int, int gc, int c) {
        return ld_w<NT>((const unsigned long long *)(W + ((size_t)gc * K + k0) / 2 + c));
      });
      mma_slab();
    }
  }

  const int n = n0 + wave * 16 + col;

  // At DKS==1 the accumulator already holds the whole K range, so the partial buffer, the
  // threadfence, the atomic and the reduction pass are all pure overhead -- and that overhead is
  // not small. Partial traffic is DKS*M*N floats: constant in the weight stream but LINEAR IN M,
  // 11.1 MB against gate_up's 44.6 MB of weights at M=40 and 17.8 MB at M=64. Picking DKS by shape
  // and by M is the launcher's job (split_k_for below); this is the path that makes DKS==1 free.
  // Bit-identical to the DKS==1 reduction path, which summed a single term -- the DIRECT=false
  // instantiation exists so that equivalence can be tested rather than asserted.
  if constexpr (DKS == 1 && DIRECT) {
    if (n < N) {
      const float rf = __int_as_float((int)Wref[n] << 23);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kb8 + e;
          if (m < M) C[(size_t)m * N + n] = (__bf16)(acc[i][e] * rf * As[m]);
        }
    }
    return;
  }

  if (n < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * M + m) * N + n] = acc[i][e];
      }
  }

  // Fused reduction. The DKS blocks covering one n-range race on an atomic counter and whoever
  // arrives last reduces in place, so there is no second kernel launch -- 13,376 of them per
  // decode run in the profile, at a 4.5 us mean host gap. It is also faster in pure kernel time
  // (-5.5% on gate_up to -29% on gdn.out) because the partials are still in cache when the
  // reduction reads them. The last block resets the counter, leaving it zeroed for the next
  // launch; launches on one stream are serialised, so no separate clearing pass is needed.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.x] = 0;

  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += DNTHREADS) {
    const float rf = __int_as_float((int)Wref[nn] << 23);
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__bf16)(sum * rf * As[m]);
    }
  }
}

// Split-K partial slab. PyTorch owns the memory; we only hold the pointer.
//
// This used to hipMalloc here and that was wrong twice over. Allocating lazily from launch() puts
// the allocation inside CUDA-graph capture whenever the torch.compile cache is warm (vLLM then
// skips the eager profile run), where it is illegal. And allocating from this .so at all produced
// a completely misleading error: quark's TileLang registers a pybind11 exception translator in the
// same ABI domain, so ANY C++ exception escaping our module is reported as
// "HIP runtime library (libamdhip64.so) not found ... TileLang's ROCm backend", pointing at our own
// pybind frame. Letting torch allocate it removes both problems and makes the buffer visible from
// Python, where its lifetime is obvious.
//
// Sized DEC_KS(4) x 64 rows x N(<=32768) x 4 B = 32 MiB, allocated in radiance_mxfp4.py. 64 rows
// because a dflash drafter at SPEC=7 reaches M = max_num_seqs x 8 = 64. Single-stream decode per
// rank is assumed. split_k_for() may choose a narrower split, which only shrinks the requirement.
// 36864, not 32768 (2026-09-16): TP=1 gate_up is N=34816, and at 32768 the gate below sent every
// decode call of the widest linear to the folded prefill tile. TP=2's widest shape is 17408, so
// for it this is a comparison against a bigger constant and nothing else; the scratch that backs
// DKS>1 is sized in radiance_mxfp4.py from the rank's own width, and nblk >= 110 picks DKS=1
// (no scratch) for the 34816 shape anyway.
#define DEC_MAX_N 36864
#define DEC_KS 4
#define DEC_DWN 8           // waves per decode block; the N tile is DEC_DWN*16 = 128 wide

// Split-K width: the smallest split whose grid still fills the machine.
//
// The partial buffer costs DKS*M*N floats of write-then-read traffic -- invisible next to the
// weight stream at M=5, dominant at M=64 (on gate_up, 17.8 MB against 44.6 MB of weights) -- and
// the in-place reduction serialises onto whichever of the DKS blocks finishes last while the other
// DKS-1 have exited. The benefit is grid fill: with DTM=1 the independent N-fragments number N/16,
// so a narrow shape cannot fill the GPU without splitting K. Cost rises with M, benefit does not,
// so the fill target the split has to clear FALLS as M rises. That is the whole rule.
//
// Measured cell by cell in ~/mxfp4_work/tier6 (mxks.hip, NCOPY=6 so every shape is DRAM-fed -- at
// NCOPY=3 the 22.3 MB down weight sits in the 64 MB Infinity Cache, reads 847 GB/s, and flatters
// whichever split reloads it least). Best DKS, us, this box:
//
//   shape     N      nblk   M=5  M=8  M=16 M=24 M=32 M=40 M=48 M=56 M=64
//   gate_up   17408   136    4    4    1    1    1    1    1    1    1
//   n8192      8192    64    4    2    2    2    2    2    2    2    2
//   n7168      7168    56    4    2    2    2    2    2    2    2    2
//   down       5120    40    4    4    4    4    4    4    4    2    2
//   out        5120    40    4    4    4    4    4    4    4    2    2
//   gate48       48     1    4    4    4    4    4    4    4    4    4
//
// Those six ARE the production shape list, read off a running server with RADIANCE_MXFP4_MHIST
// rather than derived from the model config -- the config-derived guess had the fused qkv at
// N=4096 and missed 7168 and 8192 completely, which is exactly the band where DKS=2 lives. Deriving
// this shape list by hand got it wrong; do not do that again.
//
// Two further things the table says that a coarser rule gets wrong:
//   * At M<=8 EVERY shape wants the widest split, gate_up included (96.3 us at DKS=4 against 98.0
//     at DKS=1). That is the single-stream shape -- dflash SPEC=7 makes M = num_seqs*8 -- so a rule
//     that reaches for DKS=1 on width alone would trade latency on the commonest call there is.
//   * nblk=40 crosses over at M=56, but nblk=56 and nblk=64 cross over at M=8. The crossover is not
//     a property of M alone, which is why the threshold is on nblk*ks and not on either separately.
//
// M in production is NOT just num_seqs*(SPEC+1): the same probe shows 1, 2, 8, 9, 13, 16, 21, 24,
// 32, 47, 48 and 64 all reaching this kernel, plus 72 and 99 from mixed prefill+decode batches that
// exceed DEC_MAX_TM*16 and fall through to the folded tile.
// BK for the decode tile, keyed to DTM.
//
// tier5 picked BK=128 for the whole decode band, but measured it only at M<=16 (DTM=1) and against
// a SINGLE weight buffer -- 44.6 MB of gate_up sits inside the 64 MB last-level cache, and tier5's
// own probe read 1251 GB/s there against a 635 GB/s cold-DRAM peak. Re-measured DRAM-fed
// (~/mxfp4_work/tier6/grid.hip), BK=128 still wins on geomean, but it loses hard at DTM=4 because
// LDS/block is (16*DTM + DWN*16)*(DBK+8): it grows 19.6 -> 26.1 KB across DTM while BK=64 goes only
// 10.4 -> 13.8, so at DTM=4 it is 2 resident blocks per CU against 4, occupancy 8 against 12. That
// is the same LDS cliff that set the PREFILL tile to BK=64; it reaches decode once the batch is
// deep enough. gate_up at M=56/64 measures 0.80x / 0.79x, reproduced across three separate stages.
//
// Deliberately narrow. At DTM=4 the fill rule below yields ks==1 only for gate_up, so this touches
// exactly the one shape where the win is decisive. down and out also measure ~0.955x at BK=64, and
// n8192/n7168/gate48 measure 1.00-1.04x -- all inside the 1-4% run-to-run spread seen between
// sweep stages, so they keep BK=128 until that is re-measured rather than widening a rule fitted to
// six points. Note the two axes are coupled BOTH ways: at BK=64, down and out want KS=4 where at
// BK=128 they want KS=2, so split_k_for's high-M band is a BK=128 property, not a shape property.
//
// RADIANCE_MXFP4_DECODE_BK=128 pins the old behaviour; that is the A/B control.
static bool decode_bk64(int tm, int ks) {
  static const bool pin128 = [] {
    const char *e = getenv("RADIANCE_MXFP4_DECODE_BK");
    return e && atoi(e) == 128;
  }();
  return !pin128 && tm == 4 && ks == 1;
}

static int split_k_for(int nblk, int M) {
  // Cached, not a getenv per call: this runs once per decode GEMM and there are 304 of them per
  // forward, against a 4.5 us launch gap. Same pattern as the other knobs in launch().
  static const int forced = [] {
    const char *e = getenv("RADIANCE_MXFP4_DECODE_KS");
    return e ? atoi(e) : 0;
  }();
  if (forced) return forced;
  // Fill target in blocks. Re-measured with branchless staging (~/mxfp4_work/tier7, 2026-08-28):
  // the old M<=8 special case (widest split always) INVERTED -- the serialised staging round
  // trips were what a wider split's extra fill was paying for, and with them gone the smallest
  // filling split wins at M<=8 on every shape (gate_up dks4->dks1 -2.7/-5.3%, n8192/n7168
  // dks4->dks2 -4.3/-8.4%). The down/out crossover to dks2 also moved down, M>=56 -> M>=32
  // (down M=40 61.7->59.3, out M=48 32.6->30.4). fill=110 at M<=24 and 78 above reproduces the
  // best measured cell on all six shapes within the 1-4% run spread; out M=16 is why the
  // boundary cannot drop below 24 (dks2 there is +6%).
  const int fill = (M <= 24) ? 110 : 78;
  for (int ks = 1; ks < DEC_KS; ks <<= 1)
    if (nblk * ks >= fill) return ks;
  return DEC_KS;
}
#define DEC_MAX_TM 8        // covers M<=128: a dflash drafter at SPEC=7 and max_num_seqs=16
                            // reaches batch x 8 = 128. Extended from 4 on 2026-08-29 for the
                            // 16-concurrent serve; the launcher's decode_max_m env still gates
                            // WHICH M take this kernel (64 at max_num_seqs=8, unchanged), so
                            // the extension is inert until the serve opts in. The split-K
                            // scratch must cover DEC_KS*decode_max_m rows -- radiance_mxfp4.py
                            // sizes it from RADIANCE_MXFP4_DECODE_MAX_M.
static float *g_decode_partial = nullptr;
static size_t g_decode_partial_bytes = 0;

static int *g_decode_cnt = nullptr;

static void set_decode_scratch(uintptr_t ptr, size_t bytes, uintptr_t cnt) {
  g_decode_partial = (float *)ptr;
  g_decode_partial_bytes = bytes;
  g_decode_cnt = (int *)cnt;        // must be ZEROED by the caller, once
}

static void launch_impl(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                        uintptr_t c, int M, int N, int K, int pb1, int pb2, long astride,
                        uintptr_t stream) {
  if (K % BK) throw std::runtime_error("radiance_mxfp4_fp8: K must be a multiple of 64");
  // partition boundaries must be n-tile aligned for every kernel below (decode 128, folded 64/128)
  if ((pb1 < N && pb1 % 128) || (pb2 < N && pb2 % 128))
    throw std::runtime_error("radiance_mxfp4_fp8: partition boundaries must be multiples of 128");
  if (!wref && pb1 < N) throw std::runtime_error("radiance_mxfp4_fp8: partitions need the folded path");
  dim3 block(NWAVE * 32);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  // the tuned folded kernel wins at every shape measured, so it is the default whenever the
  // per-row reference exponent is available
  if (wref) {
    static const int tn4_min_m = [] {
      const char *e = getenv("RADIANCE_MXFP4_TN4_MIN_M");
      return e ? atoi(e) : 2048;
    }();
    // Decode band. Default 0 = dark; the branch must come first because 16 < tn4_min_m.
    static const int decode_max_m = [] {
      const char *e = getenv("RADIANCE_MXFP4_DECODE_MAX_M");
      return e ? atoi(e) : 0;
    }();
    // Weight layout, set once for the process by the loader: it decides whether the tensor handed
    // to us is the checkpoint's [N, K/2] or fragment order. Both kernels below read it, and it
    // MUST agree with radiance_mxfp4.py's WPERM -- disagreeing reads the weight as garbage.
    static const bool dec_wperm = [] {
      const char *e = getenv("RADIANCE_MXFP4_WPERM");
      return e && atoi(e) != 0;
    }();
    // Epilogue fast path. Default ON (it is bit-identical and worth 7.7-8.0% at prefill shapes),
    // but reachable so the batched-throughput A/B can be run against the SAME binary and server
    // config -- comparing against an older BetterBench run would confound it with every other
    // launcher knob that has changed since.
    static const bool epifast = [] {
      const char *e = getenv("RADIANCE_MXFP4_EPIFAST");
      return !e || atoi(e) != 0;
    }();
    // Streaming weight loads in the decode kernel (see ld_w). Default off until measured.
    static const bool dec_nt = [] {
      const char *e = getenv("RADIANCE_MXFP4_DECODE_NT");
      return e && atoi(e) != 0;
    }();
    dim3 fblock(NTHREADS);
    constexpr int DWN = DEC_DWN;                 // tuned: ~/mxfp4_work/tier5 + tier6/grid.hip
    constexpr int DBND = DWN * 16;
    const int dec_nblk = (N + DBND - 1) / DBND;
    // M <= 64 keeps the shipped policy verbatim (split_k_for + decode_bk64) -- the 16-concurrent
    // extension must not move a single cell of the 8-and-under band. M in (64, 128] is a NEW
    // band, measured cell-by-cell in tier7 (2026-08-29, DRAM-fed, all six shapes to M=128):
    //   nblk >= 110 (gate_up)  -> dks1/bk64   (bk128 is +38% at M=128: 251 vs 181 us)
    //   nblk >= 48  (n8192/7168)-> dks1/bk128 (the old fill=78 rule picked dks2, -5%)
    //   nblk <= 8   (gate48)   -> dks4/bk128  (widest split, as everywhere)
    //   nblk ~40, K >= 6144 (down: 68 k-slabs) -> M <= 80: dks2/bk128; else dks4/bk64
    //     (down goes non-monotonic: at M >= 96 the wide split + BK=64's occupancy wins,
    //      d4/b64 111.8 vs d2/b128 153.4 at M=128)
    //   nblk ~40, K <  6144 (out: 24 k-slabs)  -> dks1/bk128 (few slabs, splits never pay)
    int dks, bkx;
    if (M > 64) {
      if (dec_nblk >= 110)      { dks = 1; bkx = 64; }
      else if (dec_nblk >= 48)  { dks = 1; bkx = 128; }
      else if (dec_nblk <= 8)   { dks = DEC_KS; bkx = 128; }
      else if (K >= 6144)       { if (M <= 80) { dks = 2; bkx = 128; }
                                  else         { dks = DEC_KS; bkx = 64; } }
      else                      { dks = 1; bkx = 128; }
    } else {
      dks = split_k_for(dec_nblk, M);
      bkx = 0;   // resolved below from decode_bk64, exactly as shipped
    }
    // The decode kernel stages whole DBK-byte K slabs with no k bound (stage_a / stage_w read
    // A + row*K + k0 + c across the slab), so K must be a multiple of the slab. Every stock shape
    // is a multiple of 128 and never met this; the TP=3 dummy-head padding (radiance_tp3pad)
    // makes down_proj K = 17472 / 5824 (TP=1 / TP=3): a multiple of 64 but not 128, and at
    // BK=128 the last slab read the NEXT row's bytes -- a serve that emitted nothing but token 0
    // (2026-09-06). Take the BK=64 instantiation for such a K; it exists for split 1 and 4 only,
    // so the split-2 cell widens to 4. Inert when K % 128 == 0 (all stock shapes).
    if (K % 128) {
      bkx = 64;
      if (dks == 2) dks = DEC_KS;
    }
    // DKS==1 writes C directly and needs neither the partial buffer nor the block counter;
    // anything wider needs both. Sizing the guard by the CHOSEN ks (not the widest) also means a
    // shape that would have overflowed the scratch at DKS=4 can still take the kernel at DKS=1.
    const bool have_scratch = g_decode_partial && g_decode_cnt &&
        (size_t)dks * M * N * sizeof(float) <= g_decode_partial_bytes;
    if (M > 0 && M <= decode_max_m && M <= DEC_MTILE * DEC_MAX_TM && N <= DEC_MAX_N &&
        (dks == 1 || have_scratch)) {
      float *P = g_decode_partial;
      const dim3 dblock(DWN * 32);
      const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;   // smallest tile that covers M
      if (bkx == 0) bkx = decode_bk64(tm, dks) ? 64 : 128;
#define RAD_DEC_LAUNCH(BK_, KS_, TM_)                                                         \
      hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_decode<DWN, BK_, KS_, TM_, RAD_WPERM, true,  \
                                                          RAD_NT>),                           \
                         dim3(dec_nblk, 1, KS_), dblock,                                       \
                         0, (hipStream_t)stream, (const unsigned char *)a,                     \
                         (const unsigned char *)w, (const unsigned char *)ws,                  \
                         (const unsigned char *)wref, (const float *)as, P, g_decode_cnt,      \
                         (__bf16 *)c, M, N, K, pb1, pb2, astride)
#define RAD_DEC_BY_TM(BK_, KS_)                                                               \
      if (tm == 1)      RAD_DEC_LAUNCH(BK_, KS_, 1);                                           \
      else if (tm == 2) RAD_DEC_LAUNCH(BK_, KS_, 2);                                           \
      else if (tm == 3) RAD_DEC_LAUNCH(BK_, KS_, 3);                                           \
      else if (tm == 4) RAD_DEC_LAUNCH(BK_, KS_, 4);                                           \
      else if (tm == 5) RAD_DEC_LAUNCH(BK_, KS_, 5);                                           \
      else if (tm == 6) RAD_DEC_LAUNCH(BK_, KS_, 6);                                           \
      else if (tm == 7) RAD_DEC_LAUNCH(BK_, KS_, 7);                                           \
      else              RAD_DEC_LAUNCH(BK_, KS_, 8)
#define RAD_DEC_BY_KS                                                                         \
      if (dks <= 1)      { if (bkx == 64) { RAD_DEC_BY_TM(64, 1); }                            \
                           else           { RAD_DEC_BY_TM(128, 1); } }                         \
      else if (dks == 2) { RAD_DEC_BY_TM(128, 2); }                                            \
      else               { if (bkx == 64) { RAD_DEC_BY_TM(64, 4); }                            \
                           else           { RAD_DEC_BY_TM(128, 4); } }
      // Streaming (NT) weight loads pay ONLY under WPERM, where a wave's weight read is one
      // whole 128 B line: measured 2026-09-01 (tier7 ntab, DRAM-fed) -5..-8% on gate_up /
      // n8192 / n7168 / down at M<=48, ~0 at M=64, i.e. 97% of the 635 GB/s stream. On the
      // checkpoint layout a row's slab is half (DBK=128) or a quarter (64) of a line and the
      // rest is only wanted by the NEXT slab, so NT re-fetches every line: 2.0-3.6x SLOWER.
      // Two restagings that batch a thread's loads for the whole line (8 B x G, then one 16 B
      // per lane / 4 rows per wave) were STILL 1.4-1.8x under NT and -5..-19% without it, so
      // the row layout keeps plain loads and dec_nt is ignored there.
      if (dec_wperm) {
#define RAD_WPERM true
        if (dec_nt) {
#define RAD_NT true
          RAD_DEC_BY_KS;
#undef RAD_NT
        } else {
#define RAD_NT false
          RAD_DEC_BY_KS;
#undef RAD_NT
        }
#undef RAD_WPERM
      } else {
#define RAD_WPERM false
#define RAD_NT false
        RAD_DEC_BY_KS;
#undef RAD_NT
#undef RAD_WPERM
      }
#undef RAD_DEC_BY_KS
#undef RAD_DEC_BY_TM
#undef RAD_DEC_LAUNCH
      return;
      // no scratch (alloc failed) -- fall through to the folded kernel, never produce nothing
    }
#define RAD_FOLDED_E(TN_, WP_, EF_)                                                             \
    do {                                                                                        \
      constexpr int B_ = BNF_OF(TN_);                                                           \
      dim3 fgrid((N + B_ - 1) / B_, (M + BMF - 1) / BMF);                                       \
      hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_folded<TN_, WP_, EF_>), fgrid, fblock, 0,     \
                         (hipStream_t)stream, (const unsigned char *)a,                         \
                         (const unsigned char *)w, (const unsigned char *)ws,                   \
                         (const unsigned char *)wref, (const float *)as, (__bf16 *)c, M, N, K, \
                         pb1, pb2, astride);                                                    \
    } while (0)
#define RAD_FOLDED(TN_, WP_)                                                                    \
    do { if (epifast) RAD_FOLDED_E(TN_, WP_, true); else RAD_FOLDED_E(TN_, WP_, false); } while (0)
    // TN=4 only when the shape is wide enough to use the 128-wide tile: at N=48 (gdn gate)
    // TN=4 computes 128 columns against 48 real ones and measures 1.8-2x SLOWER than TN=2 at
    // every M >= 2048 (56 vs 101-115 us, ~/mxfp4_work/tier7 tnx). The M crossover itself
    // re-measured within noise of 2048 on all large-N shapes with branchless staging.
    if (M >= tn4_min_m && N >= BNF_OF(4)) {
      if (dec_wperm) RAD_FOLDED(4, true); else RAD_FOLDED(4, false);
    } else {
      if (dec_wperm) RAD_FOLDED(2, true); else RAD_FOLDED(2, false);
    }
#undef RAD_FOLDED
#undef RAD_FOLDED_E
  } else
    hipLaunchKernelGGL(radiance_mxfp4_fp8_gemm, grid, block, 0, (hipStream_t)stream,
                       (const unsigned char *)a, (const unsigned char *)w,
                       (const unsigned char *)ws, (const float *)as, (__bf16 *)c, M, N, K);
}

// Stock entry: one activation, one partition (bit-identical to before the partition select).
static void launch(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                   uintptr_t c, int M, int N, int K, uintptr_t stream) {
  launch_impl(a, w, ws, wref, as, c, M, N, K, 1 << 30, 1 << 30, 0, stream);
}
// Merged-linear entry (paroquant): a = [P, M, K] rotated copies astride bytes apart, as = [P, M];
// n-blocks in [0, pb1) read copy 0, [pb1, pb2) copy 1, [pb2, N) copy 2.
static void launch_p(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                     uintptr_t c, int M, int N, int K, int pb1, int pb2, long astride,
                     uintptr_t stream) {
  launch_impl(a, w, ws, wref, as, c, M, N, K, pb1, pb2, astride, stream);
}


// ---- fused residual-add + Gemma RMSNorm + per-token fp8 quant -------------------------------
//
// The decode-step epilogue of every RowParallel linear: after the all-reduce, the traced graph
// runs (add+rms_norm) and the traced per-token quant -- 2-3 inductor kernels per site whose
// tiling inductor chooses for the M=8192 compile hint and replays at M=8 (XBLOCK=1, one row per
// workgroup, ~4.4 us of kernels + 2 dispatch gaps per site). This kernel does the whole epilogue
// in ONE launch. Semantics mirror vllm.ir.ops.fused_add_rms_norm with the GemmaRMSNorm weight
// convention plus the radiance traced quant, in exactly its precision:
//   s      = f32(y) + f32(residual)         (fp32 sum feeds BOTH residual_out and the variance)
//   res_o  = bf16(s)
//   n      = bf16( s * rsqrt(mean(s^2)+eps) * (f32(w)+1) )   <- Gemma: weight is (1+w), fp32
//   scale  = max(amax|f32(n)|/448, 1/(448*512))               <- amax over the BF16-ROUNDED n
//   q      = e4m3_rne(clamp(f32(n)/scale, +-448))
// One workgroup per row; latency-bound at decode M<=8 like everything else in the step, and at
// prefill M=chunk there are thousands of rows so occupancy takes care of itself.
#define ARNQ_T 256
typedef unsigned int uint32_tt;
union ArnqVec8 { uint4 u4; __bf16 h[8]; };
template <int ART, bool TILED = false>
__global__ __launch_bounds__(ART) void radiance_add_rms_quant(
    const __bf16 *__restrict__ y, const __bf16 *__restrict__ res, const __bf16 *__restrict__ w,
    unsigned char *__restrict__ q, float *__restrict__ scale, __bf16 *__restrict__ res_out,
    int K, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  y += (long)row * K; res += (long)row * K; res_out += (long)row * K;
  if (!TILED) q += (long)row * K;
  // TILED: q is the fragment-tiled layout the atiled GEMM reads (see there): 8-group g of this
  // row lands in fragment (row/16, g/2) at lane (row%16) + 16*(g%2). Same single uint2 store;
  // the scatter across fragments measured +0.3..+3.8% on this kernel (tier7/aq.hip).
  const int mt = row >> 4, lrow = row & 15, kst = K >> 4;
  // 16-byte vectorized: thread t owns 8-element groups t, t+T, t+2T, ... The scalar version of
  // this kernel measured 12.5 us at M=8 under graph replay -- 20 independent 2-byte loads per
  // thread per tensor is pure latency serialization at 8 workgroups. Vectorized it is one
  // uint4 load per tensor per group. K must be a multiple of 8 (5120/8704/2048 all are).
  constexpr int MAXG = 5;                       // 8704 / (256*8) rounds up to 5
  const int KV = K >> 3;                        // groups of 8
  float sv[MAXG][8];
  int ng = 0;
  float ssq = 0.f;
  for (int g = tid; g < KV; g += ART, ++ng) {
    ArnqVec8 vy, vr;
    vy.u4 = ((const uint4 *)y)[g];
    vr.u4 = ((const uint4 *)res)[g];
    ArnqVec8 vo;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float v = (float)vy.h[j] + (float)vr.h[j];
      sv[ng][j] = v;
      vo.h[j] = (__bf16)v;
      ssq += v * v;
    }
    ((uint4 *)res_out)[g] = vo.u4;
  }
  __shared__ float lds[ART / 32];
  for (int o = 16; o; o >>= 1) ssq += __shfl_down(ssq, o);
  if ((tid & 31) == 0) lds[tid >> 5] = ssq;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v += __shfl_down(v, o);
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(lds[0] / (float)K + eps);
  __syncthreads();                       // lds[0] is reused for the amax reduction below
  float amax = 0.f;
  int mg = 0;
  for (int g = tid; g < KV; g += ART, ++mg) {
    ArnqVec8 vw;
    vw.u4 = ((const uint4 *)w)[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float nv = (float)(__bf16)(sv[mg][j] * inv * ((float)vw.h[j] + 1.f));
      sv[mg][j] = nv;
      amax = fmaxf(amax, fabsf(nv));
    }
  }
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[row] = sc;
  mg = 0;
  for (int g = tid; g < KV; g += ART, ++mg) {
    uint32_tt lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    if (TILED) ((uint2 *)q)[(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
    else ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}

// ---- fused SiLU-mul + per-token fp8 quant ----------------------------------------------------
// The MLP's activation epilogue: gate_up output [M, 2N] -> silu(gate) * up -> per-token e4m3.
// Same skeleton and precision contract as radiance_add_rms_quant: every intermediate is rounded
// exactly where the traced path rounds (silu to bf16, product to bf16, amax over the rounded
// product). One workgroup per row.
template <int ART, bool TILED = false>
__global__ __launch_bounds__(ART) void radiance_silu_mul_quant(
    const __bf16 *__restrict__ gu, unsigned char *__restrict__ q, float *__restrict__ scale,
    int N) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  gu += (long)row * 2 * N;
  if (!TILED) q += (long)row * N;
  const int mt = row >> 4, lrow = row & 15, kst = N >> 4;   // TILED: as in add_rms_quant
  constexpr int MAXG = 5;                  // N <= 10240 (8704 in this model)
  const int NV = N >> 3;
  float pv[MAXG][8];
  int ng = 0;
  float amax = 0.f;
  for (int g = tid; g < NV; g += ART, ++ng) {
    ArnqVec8 vg, vu;
    vg.u4 = ((const uint4 *)gu)[g];
    vu.u4 = ((const uint4 *)(gu + N))[g];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float x = (float)vg.h[j];
      float t = (float)(__bf16)(x / (1.f + expf(-x)));       // silu, rounded to bf16
      float p = (float)(__bf16)(t * (float)vu.h[j]);         // product, rounded to bf16
      pv[ng][j] = p;
      amax = fmaxf(amax, fabsf(p));
    }
  }
  __shared__ float lds[ART / 32];
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[row] = sc;
  int mg = 0;
  for (int g = tid; g < NV; g += ART, ++mg) {
    uint32_tt lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      float a = fminf(fmaxf(pv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(pv[mg][j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      float a = fminf(fmaxf(pv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(pv[mg][j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    if (TILED) ((uint2 *)q)[(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)] = make_uint2(lo, hi);
    else ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}

static void launch_silu_mul_quant(uintptr_t gu, uintptr_t q, uintptr_t scale, int M, int N,
                                  uintptr_t stream, int tiled) {
  if (N > 40 * 512 || (N & 7))
    throw std::runtime_error("radiance_silu_mul_quant: N must be 8-aligned and <= 20480");
  if (tiled && (N & 15)) throw std::runtime_error("radiance_silu_mul_quant: tiled needs N % 16 == 0");
  // Above the 256-thread block's register budget (MAXG=5 groups of 8 -> N <= 10240, which is
  // every TP>=2 shape) the 512-thread instantiation covers N <= 20480: TP=1's intermediate is
  // 17408 (2026-09-16). Shapes at or below 10240 launch exactly as before.
  if (N > 40 * ARNQ_T) {
    if (tiled)
      hipLaunchKernelGGL((radiance_silu_mul_quant<512, true>), dim3(M), dim3(512), 0,
                         (hipStream_t)stream, (const __bf16 *)gu, (unsigned char *)q, (float *)scale, N);
    else
      hipLaunchKernelGGL((radiance_silu_mul_quant<512>), dim3(M), dim3(512), 0, (hipStream_t)stream,
                         (const __bf16 *)gu, (unsigned char *)q, (float *)scale, N);
    return;
  }
  if (tiled) {
    if (M >= 2048)
      hipLaunchKernelGGL((radiance_silu_mul_quant<512, true>), dim3(M), dim3(512), 0,
                         (hipStream_t)stream, (const __bf16 *)gu, (unsigned char *)q, (float *)scale, N);
    else
      hipLaunchKernelGGL((radiance_silu_mul_quant<ARNQ_T, true>), dim3(M), dim3(ARNQ_T), 0,
                         (hipStream_t)stream, (const __bf16 *)gu, (unsigned char *)q, (float *)scale, N);
    return;
  }
  if (M >= 2048)
    hipLaunchKernelGGL((radiance_silu_mul_quant<512>), dim3(M), dim3(512), 0, (hipStream_t)stream,
                       (const __bf16 *)gu, (unsigned char *)q, (float *)scale, N);
  else
    hipLaunchKernelGGL((radiance_silu_mul_quant<ARNQ_T>), dim3(M), dim3(ARNQ_T), 0,
                       (hipStream_t)stream, (const __bf16 *)gu, (unsigned char *)q,
                       (float *)scale, N);
}

// ---- fused per-head gated RMSNorm + per-token fp8 quant (GDN out_proj input) ----------------
// Replaces RMSNormGated(norm_before_gate=True, group per 128-wide head) + the traced per-token
// quant in _output_projection, which inductor emits as two kernels per GDN layer (a variance
// reduction and a normalize+gate+quant). Precision contract mirrors radiance_add_rms_quant and
// the traced reference: fp32 throughout, ((x * rsqrt(mean(x^2) + eps)) * w) * silu(z) rounded
// to bf16 once, amax over the rounded values, e4m3 with the vLLM min-scale clamp. The per-head
// sum of squares is a 16-lane xor-shuffle: 16 consecutive 8-groups make one head and, with ART a
// multiple of 16 and N a multiple of 128, every 16-lane segment is either fully active or fully
// idle in each pass. x: [M, N] contiguous bf16; z: [M, N] bf16 with row stride zs (a column
// slice of the fused qkvz projection, no copy); w: [128] bf16 shared by all heads.
template <int ART>
__global__ __launch_bounds__(ART) void radiance_gdn_norm_quant(
    const __bf16 *__restrict__ x, const __bf16 *__restrict__ z, long zs,
    const __bf16 *__restrict__ w, unsigned char *__restrict__ q, float *__restrict__ scale,
    int N, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += (long)row * N; z += (long)row * zs; q += (long)row * N;
  constexpr int MAXG = 5;                       // N <= ART*8*5 = 10240 (3072 in this model)
  const int NV = N >> 3;                        // 8-element groups; 16 per head
  float sv[MAXG][8];
  int ng = 0;
  float amax = 0.f;
  for (int g = tid; g < NV; g += ART, ++ng) {
    ArnqVec8 vx, vz, vw;
    vx.u4 = ((const uint4 *)x)[g];
    float ssq = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = (float)vx.h[j];
      sv[ng][j] = v;
      ssq += v * v;
    }
    for (int o = 8; o; o >>= 1) ssq += __shfl_xor(ssq, o, 16);
    const float inv = rsqrtf(ssq * (1.f / 128.f) + eps);
    vz.u4 = ((const uint4 *)z)[g];
    vw.u4 = ((const uint4 *)w)[g & 15];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float zz = (float)vz.h[j];
      const float sg = zz / (1.f + expf(-zz));                       // silu, fp32
      const float nv = (float)(__bf16)(((sv[ng][j] * inv) * (float)vw.h[j]) * sg);
      sv[ng][j] = nv;
      amax = fmaxf(amax, fabsf(nv));
    }
  }
  __shared__ float lds[ART / 32];
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < ART / 32) {
    float v = lds[tid];
    for (int o = ART / 64; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[row] = sc;
  int mg = 0;
  for (int g = tid; g < NV; g += ART, ++mg) {
    uint32_tt lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      lo |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << (j * 8);
    }
#pragma unroll
    for (int j = 4; j < 8; j += 2) {
      float a = fminf(fmaxf(sv[mg][j] * rs, -448.f), 448.f);
      float b = fminf(fmaxf(sv[mg][j + 1] * rs, -448.f), 448.f);
      hi |= (__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu) << ((j - 4) * 8);
    }
    ((uint2 *)q)[g] = make_uint2(lo, hi);
  }
}

static void launch_gdn_norm_quant(uintptr_t x, uintptr_t z, long zs, uintptr_t w, uintptr_t q,
                                  uintptr_t scale, int M, int N, double eps, uintptr_t stream) {
  if (N > 40 * ARNQ_T || (N & 127))
    throw std::runtime_error("radiance_gdn_norm_quant: N must be a multiple of 128 and <= 10240");
  if (zs & 7) throw std::runtime_error("radiance_gdn_norm_quant: z row stride must be 8-aligned");
  hipLaunchKernelGGL((radiance_gdn_norm_quant<ARNQ_T>), dim3(M), dim3(ARNQ_T), 0,
                     (hipStream_t)stream, (const __bf16 *)x, (const __bf16 *)z, zs,
                     (const __bf16 *)w, (unsigned char *)q, (float *)scale, N, (float)eps);
}

static void launch_add_rms_quant(uintptr_t y, uintptr_t res, uintptr_t w, uintptr_t q,
                                 uintptr_t scale, uintptr_t res_out, int M, int K, double eps,
                                 uintptr_t stream, int tiled) {
  if (K > 40 * ARNQ_T || (K & 7)) throw std::runtime_error("radiance_add_rms_quant: K must be 8-aligned and <= 10240");
  if (tiled) {
    if (K & 15) throw std::runtime_error("radiance_add_rms_quant: tiled needs K % 16 == 0");
    if (M >= 2048)
      hipLaunchKernelGGL((radiance_add_rms_quant<512, true>), dim3(M), dim3(512), 0, (hipStream_t)stream,
                         (const __bf16 *)y, (const __bf16 *)res, (const __bf16 *)w,
                         (unsigned char *)q, (float *)scale, (__bf16 *)res_out, K, (float)eps);
    else
      hipLaunchKernelGGL((radiance_add_rms_quant<ARNQ_T, true>), dim3(M), dim3(ARNQ_T), 0,
                         (hipStream_t)stream, (const __bf16 *)y, (const __bf16 *)res,
                         (const __bf16 *)w, (unsigned char *)q, (float *)scale,
                         (__bf16 *)res_out, K, (float)eps);
    return;
  }
  // T=512 at prefill-class M: measured 726 -> 685 us (405 -> 429 GB/s) at M=8192 on a clean
  // GPU. Changes the per-row summation order (512-way vs 256-way striding) => gated by GSM8K,
  // not byte-compare. Decode-class M keeps T=256 (T=512 measured worse at small M in the nq
  // sweep). The PREFETCH variant this replaces measured 828 us at M=8192 -- 14% WORSE than the
  // original -- its +40 VGPRs/thread costs occupancy exactly when 8192 blocks compete; the
  // M=2048 idle-GPU microbench that justified it was the fill-the-GPU-first trap.
  if (M >= 2048)
    hipLaunchKernelGGL((radiance_add_rms_quant<512>), dim3(M), dim3(512), 0, (hipStream_t)stream,
                       (const __bf16 *)y, (const __bf16 *)res, (const __bf16 *)w,
                       (unsigned char *)q, (float *)scale, (__bf16 *)res_out, K, (float)eps);
  else
    hipLaunchKernelGGL((radiance_add_rms_quant<ARNQ_T>), dim3(M), dim3(ARNQ_T), 0,
                       (hipStream_t)stream, (const __bf16 *)y, (const __bf16 *)res,
                       (const __bf16 *)w, (unsigned char *)q, (float *)scale,
                       (__bf16 *)res_out, K, (float)eps);
}

// GEMM on a fragment-tiled activation (see radiance_mxfp4_fp8_gemm_atiled). The caller decides
// the layout per tensor (radiance_mxfp4.py a_tiled_*); this only dispatches. Prefill-class M
// only: the decode kernel reads row-major A, so the Python side never tiles below its band.
static void launch_at_impl(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                           uintptr_t c, int M, int N, int K, int pb1, int pb2, long astride,
                           uintptr_t stream) {
  if (K % 128) throw std::runtime_error("radiance_mxfp4_fp8: tiled A needs K % 128 == 0");
  if ((pb1 < N && pb1 % 128) || (pb2 < N && pb2 % 128))
    throw std::runtime_error("radiance_mxfp4_fp8: partition boundaries must be multiples of 128");
  if (!wref) throw std::runtime_error("radiance_mxfp4_fp8: tiled A needs the folded path (wref)");
  static const int tn4_min_m = [] {
    const char *e = getenv("RADIANCE_MXFP4_TN4_MIN_M");
    return e ? atoi(e) : 2048;
  }();
  static const bool wperm = [] {
    const char *e = getenv("RADIANCE_MXFP4_WPERM");
    return e && atoi(e) != 0;
  }();
  dim3 fblock(NTHREADS);
#define RAD_AT(TN_, WP_)                                                                        \
  do {                                                                                          \
    constexpr int B_ = BNF_OF(TN_);                                                             \
    dim3 fgrid((N + B_ - 1) / B_, (M + BMF - 1) / BMF);                                         \
    hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_atiled<TN_, WP_, (TN_ == 2 ? 128 : 64)>),      \
                       fgrid, fblock, 0,                                                        \
                       (hipStream_t)stream, (const unsigned char *)a, (const unsigned char *)w, \
                       (const unsigned char *)ws, (const unsigned char *)wref, (const float *)as, \
                       (__bf16 *)c, M, N, K, pb1, pb2, astride);                               \
  } while (0)
  if (M >= tn4_min_m && N >= BNF_OF(4)) { if (wperm) RAD_AT(4, true); else RAD_AT(4, false); }
  else                                  { if (wperm) RAD_AT(2, true); else RAD_AT(2, false); }
#undef RAD_AT
}
static void launch_at(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                      uintptr_t c, int M, int N, int K, uintptr_t stream) {
  launch_at_impl(a, w, ws, wref, as, c, M, N, K, 1 << 30, 1 << 30, 0, stream);
}
static void launch_at_p(uintptr_t a, uintptr_t w, uintptr_t ws, uintptr_t wref, uintptr_t as,
                        uintptr_t c, int M, int N, int K, int pb1, int pb2, long astride,
                        uintptr_t stream) {
  launch_at_impl(a, w, ws, wref, as, c, M, N, K, pb1, pb2, astride, stream);
}

