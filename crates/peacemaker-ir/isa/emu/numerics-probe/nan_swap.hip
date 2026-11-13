// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 hardware qualification of NaN-payload operand-order behaviour for FP32 add / mul / fmac, native VOP2 and
// BOTH halves (X and Y) of the VOPD dual-issue forms (v_dual_add_f32 / v_dual_mul_f32 / v_dual_fmac_f32), original and
// src0/src1-swapped, under all 4 FP32 denorm x 4 rounding MODE settings (16 modes).
//
// This is a raw-ISA oracle: every target operation is an inline-asm instruction; hipcc only hosts the asm and the
// integer-only bookkeeping. No floating-point expression is ever compiled for a target operation.
//
// usage: nan_swap OUTDIR [SUITES]            (run.sh-compatible: BIN OUTDIR SUITES)
//   SUITES = all (default) | comma list of: nan, tri, rand, edges
//   env NS_MODES   mode list/ranges, e.g. "0-15" (default all) or "3,7,12-15"; mode m = denorm(m&3) | round(m>>2)<<..
//   env NS_RANDOM  number of deterministic random raw triples per mode/op (default 4194304, minimum 1048576)
//   env NS_SEED    64-bit seed (default 0x5eed0bad1ee70001), logged
//   env NS_CHUNK   NaN-sweep items per launch (default 2097152; bounds each launch, not the totals)
//   env NS_BUDGET_SECS  wall budget; no new (mode,suite) cell starts after it is exceeded and the run is reported
//                       incomplete (exit 4). 0 = unlimited (default)
//
// MODE: FP_ROUND single = bits [1:0] (0 RNE, 1 +inf, 2 -inf, 3 zero); FP_DENORM single = bits [5:4]
// (bit4 = allow input denormals, bit5 = allow output denormals; 0 flush in+out, 1 flush out, 2 flush in, 3 none).
// gfx12 MODE has no IEEE / DX10_CLAMP bits; only the two single-precision fields are written (never reserved bits).
// Per wave, after the write, MODE is read back (s_getreg_b32) and four VALU control instructions prove the written
// denorm/round behaviour (input-flush, output-flush, round-up, round-down controls) -- see mode_prologue().
//
// Case (a, b, c): add and mul use (a, b); fmac computes a*b + c with accumulator c. Every case evaluates six hardware
// results: native orig (a,b), native swapped (b,a), dual X orig, dual Y orig, dual X swapped, dual Y swapped.
// Dual orig runs  v_dual_OP v4,v0,v2 :: v_dual_OP v5,v1,v3  with v0=v1=a, v2=v3=b; dual swapped with v0=v1=b, v2=v3=a;
// fmac accumulators v4=v5=c in both. So both halves execute both orders on identical operands.
//
// Integer-only oracle (per order, per position): first NaN among (a,b) -> quiet(that NaN) (sign/payload preserved);
// add: inf + (-inf) -> default NaN; mul/fmac: inf * zero -> default NaN (zero includes input-flushed denormals in
// modes whose input denorm flush is on); fmac then: NaN c -> quiet(c); then infinite product vs opposite infinite c
// -> default NaN. Overflowing finite product vs infinite c is rounding-dependent: counted as unchecked, not judged.
// "Default NaN" is judged as exactly 0xffc00000 (hardware-proven sign; no 0x7fc00000, no other payload); the 7fc/ffc/other tallies stay as evidence.
//
// Counters are streamed on device (byte-packed per-thread bit-plane accumulation, wave/block reduction, one atomic per
// counter per block). No atomic per tuple. Mismatch/evidence samples: per (suite,mode,op,class) the first <=32 slots,
// claimed through a gated atomic that only runs while the class has free slots and at most once per thread per class
// per launch. Evidence samples are retained even when there are zero mismatches.
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern char** environ;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); fflush(stderr); exit(3); } } while (0)

#define NC 32      // counters per op
#define NOPS 3     // add, mul, fmac
#define NCNT (NOPS * NC)
#define NCLS 8     // sample classes
#define SLOTS 32   // samples per (suite,mode,op,class)
#define REC 16     // u32 per sample record
#define NNAN_ITEMS 16777214u   // 2 signs * (2^23 - 1) nonzero mantissas
#define NTRI 64000u            // 40^3

enum { OP_ADD = 0, OP_MUL = 1, OP_FMAC = 2 };
// counter ids (each is a per-case boolean)
enum {
  C_CASES, C_NANEXP, C_INVEXP, C_UNK, C_SEL_NO, C_SEL_NS, C_SEL_XO, C_SEL_YO, C_SEL_XS, C_SEL_YS, C_SEL_ANY,
  C_DU_XO, C_DU_YO, C_DU_XS, C_DU_YS, C_DU_ANY, C_XY_O, C_XY_S, C_SW_N, C_SW_X, C_SW_Y, C_SW_ANY,
  C_SAFE, C_SAFE_BAD, C_UNSAFE, C_UNSAFE_SW, C_BOTHNAN, C_UNEXP, C_INV7FC, C_INVFFC, C_INVOTH, C_NANRES
};
// sample classes
enum { CL_SELBAD, CL_DUALBAD, CL_SAFEBAD, CL_SWAP, CL_SELOK_C, CL_SELOK_AB, CL_UNEXP, CL_SAFE_BOTHNAN };
// oracle kinds
enum { K_NONE = 0, K_A = 1, K_B = 2, K_INVALID = 3, K_C = 4, K_INFCONF = 5, K_UNKNOWN = 6 };

#define LIST_V \
  0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff800001u, \
  0x7fbfffffu, 0xffbfffffu, 0x7fc00001u, 0xffc00001u, 0x7fffffffu, 0xffffffffu, 0x7fc0beefu, 0xffa0beefu, \
  0x3f800000u, 0xbf800000u, 0x40000000u, 0xc0000000u, 0x3f000000u, 0xbf000000u, 0x00000001u, 0x80000001u, \
  0x007fffffu, 0x807fffffu, 0x00400000u, 0x80400000u, 0x00800000u, 0x80800000u, 0x7f7fffffu, 0xff7fffffu, \
  0x7f000000u, 0xff000000u, 0x3f800001u, 0xbf800001u, 0x20000000u, 0xa0000000u, 0x5f000000u, 0xdf000000u
#define LIST_PART \
  0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x3f800000u, 0xbf800000u, 0x00000001u, 0x80000001u, \
  0x007fffffu, 0x807fffffu, 0x7f7fffffu, 0xff7fffffu
#define LIST_C_FIXED /* idx 0,1 are derived NaNs (placeholders) */ \
  0u, 0u, 0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x3f800000u, 0xff7fffffu

static const uint32_t H_V[40] = {LIST_V};
static const uint32_t H_PART[12] = {LIST_PART};
static const uint32_t H_C[8] = {LIST_C_FIXED};
__constant__ uint32_t kV[40] = {LIST_V};
__constant__ uint32_t kPart[12] = {LIST_PART};
__constant__ uint32_t kC[8] = {LIST_C_FIXED};

// ---------------------------------------------------------------- integer helpers (host+device)
__host__ __device__ inline uint32_t nzm(uint32_t m) { m &= 0x7fffffu; return m ? m : 1u; }
__host__ __device__ inline uint32_t is_nan(uint32_t x) { return (x & 0x7fffffffu) > 0x7f800000u; }
__host__ __device__ inline uint32_t is_inf(uint32_t x) { return (x & 0x7fffffffu) == 0x7f800000u; }
// zero, or (when input flush is on) a denormal
__host__ __device__ inline uint32_t is_zin(uint32_t x, bool fi) { return (x & 0x7fffffffu) == 0 || (fi && (x & 0x7f800000u) == 0); }
// derived NaN partners 12..15 of NaN a: sign flip | quiet toggle | payload change | all three (mantissa forced nonzero)
__host__ __device__ inline uint32_t part_derived(uint32_t p, uint32_t a) {
  const uint32_t s = a & 0x80000000u, m = a & 0x7fffffu;
  switch (p) {
    case 12: return a ^ 0x80000000u;
    case 13: return s | 0x7f800000u | nzm(m ^ 0x400000u);
    case 14: return s | 0x7f800000u | nzm(m ^ 0x2aaaaau);
    default: return (s ^ 0x80000000u) | 0x7f800000u | nzm(m ^ 0x400000u ^ 0x155555u);
  }
}
__host__ __device__ inline uint32_t c_derived(uint32_t idx, uint32_t m) {
  return idx == 0 ? (0x7f800000u | nzm(m ^ 0x600000u)) : (0xff800000u | nzm(m ^ 0x1a5a5au));
}
__host__ __device__ inline uint64_t mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// ---------------------------------------------------------------- MODE write / readback / controls
// ext[] (u32, per (mode, suite)): 0 waves, 1 ctl_bad_waves, 2 mode_bad_waves, 3 mode_or, 4 mode_and, 5..8 observed
// controls (in-flush, out-flush, round-up, round-down) from wave 0, 9..12 expected controls.
constexpr uint32_t exp_ctl_in(int den) { return (den & 1) ? 0x00800000u : 0u; }
constexpr uint32_t exp_ctl_out(int den) { return (den & 2) ? 0x00400000u : 0u; }
constexpr uint32_t exp_ctl_r1(int rnd) { return rnd <= 1 ? 0x3f800001u : 0x3f800000u; }  // 1 + 1.5*2^-24
constexpr uint32_t exp_ctl_r2(int rnd) { return (rnd == 0 || rnd == 2) ? 0xbf800001u : 0xbf800000u; }  // -(1 + 1.5*2^-24)

template <int M>
__device__ __forceinline__ void mode_prologue(uint32_t* ext) {
  constexpr int DEN = M & 3, RND = M >> 2;
  asm volatile(
      "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), %0\n\t"
      "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), %1\n\t"
      "s_nop 7" ::"i"(DEN), "i"(RND) : "memory");
  uint32_t raw;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 32)" : "=s"(raw)::"memory");
  uint32_t ci, co, r1, r2;
  asm volatile("v_mul_f32 %0, %1, %2" : "=v"(ci) : "v"(0x00400000u), "v"(0x40000000u));  // denorm * 2.0: input flush -> 0
  asm volatile("v_mul_f32 %0, %1, %2" : "=v"(co) : "v"(0x00800000u), "v"(0x3f000000u));  // minnorm * 0.5: output flush -> 0
  asm volatile("v_add_f32 %0, %1, %2" : "=v"(r1) : "v"(0x3f800000u), "v"(0x33c00000u));  // 1 + 1.5*2^-24
  asm volatile("v_add_f32 %0, %1, %2" : "=v"(r2) : "v"(0xbf800000u), "v"(0xb3c00000u));  // -1 - 1.5*2^-24
  const uint32_t cbad = (ci != exp_ctl_in(DEN)) | (co != exp_ctl_out(DEN)) | (r1 != exp_ctl_r1(RND)) | (r2 != exp_ctl_r2(RND));
  const uint32_t mbad = (raw & 0x33u) != ((uint32_t)RND | ((uint32_t)DEN << 4));
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&ext[0], 1u);
    if (cbad) atomicAdd(&ext[1], 1u);
    if (mbad) atomicAdd(&ext[2], 1u);
    atomicOr(&ext[3], raw);
    atomicAnd(&ext[4], raw);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      ext[5] = ci; ext[6] = co; ext[7] = r1; ext[8] = r2;
      ext[9] = exp_ctl_in(DEN); ext[10] = exp_ctl_out(DEN); ext[11] = exp_ctl_r1(RND); ext[12] = exp_ctl_r2(RND);
    }
  }
}

// ---------------------------------------------------------------- hardware operations (inline asm only)
// Native VOP2. Plain "v" operands; VOP2 reads sources before writing, so output/input aliasing is architecturally fine.
template <int OP>
__device__ __forceinline__ uint32_t nat(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  if (OP == OP_ADD) {
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
  } else if (OP == OP_MUL) {
    asm volatile("v_mul_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
  } else {
    r = c;  // accumulator
    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(r) : "v"(a), "v"(b));
  }
  return r;
}

// VOPD, both halves, both orders, in one asm block.
// Hard-coded registers v0..v5, all declared clobbered. LLVM models inline-asm clobbers as early-clobber defs, so no
// input operand (%4..%6) can be allocated into v0..v5; the outputs are "=&v" so they can alias neither inputs nor
// each other nor v0..v5, and are written only after the last read of every input they could otherwise shadow.
// Legality: vdst parity X=v4 (even) / Y=v5 (odd); src0 banks v0 (0) / v1 (1) distinct; src1 banks v2 (2) / v3 (3)
// distinct. fmac accumulators (dst) v4 / v5 both initialised to c before each dual. Swapped order is formed by
// re-copying (v0=v1=b, v2=v3=a) so X and Y each execute (a,b) and (b,a).
template <int OP>
__device__ __forceinline__ void dual(uint32_t a, uint32_t b, uint32_t c, uint32_t& xo, uint32_t& yo, uint32_t& xs, uint32_t& ys) {
  if (OP == OP_ADD) {
    asm volatile(
        "v_mov_b32 v0, %4\n\tv_mov_b32 v1, %4\n\tv_mov_b32 v2, %5\n\tv_mov_b32 v3, %5\n\t"
        "v_dual_add_f32 v4, v0, v2 :: v_dual_add_f32 v5, v1, v3\n\t"
        "v_mov_b32 %0, v4\n\tv_mov_b32 %1, v5\n\t"
        "v_mov_b32 v0, %5\n\tv_mov_b32 v1, %5\n\tv_mov_b32 v2, %4\n\tv_mov_b32 v3, %4\n\t"
        "v_dual_add_f32 v4, v0, v2 :: v_dual_add_f32 v5, v1, v3\n\t"
        "v_mov_b32 %2, v4\n\tv_mov_b32 %3, v5"
        : "=&v"(xo), "=&v"(yo), "=&v"(xs), "=&v"(ys)
        : "v"(a), "v"(b)
        : "v0", "v1", "v2", "v3", "v4", "v5");
  } else if (OP == OP_MUL) {
    asm volatile(
        "v_mov_b32 v0, %4\n\tv_mov_b32 v1, %4\n\tv_mov_b32 v2, %5\n\tv_mov_b32 v3, %5\n\t"
        "v_dual_mul_f32 v4, v0, v2 :: v_dual_mul_f32 v5, v1, v3\n\t"
        "v_mov_b32 %0, v4\n\tv_mov_b32 %1, v5\n\t"
        "v_mov_b32 v0, %5\n\tv_mov_b32 v1, %5\n\tv_mov_b32 v2, %4\n\tv_mov_b32 v3, %4\n\t"
        "v_dual_mul_f32 v4, v0, v2 :: v_dual_mul_f32 v5, v1, v3\n\t"
        "v_mov_b32 %2, v4\n\tv_mov_b32 %3, v5"
        : "=&v"(xo), "=&v"(yo), "=&v"(xs), "=&v"(ys)
        : "v"(a), "v"(b)
        : "v0", "v1", "v2", "v3", "v4", "v5");
  } else {
    asm volatile(
        "v_mov_b32 v0, %4\n\tv_mov_b32 v1, %4\n\tv_mov_b32 v2, %5\n\tv_mov_b32 v3, %5\n\t"
        "v_mov_b32 v4, %6\n\tv_mov_b32 v5, %6\n\t"
        "v_dual_fmac_f32 v4, v0, v2 :: v_dual_fmac_f32 v5, v1, v3\n\t"
        "v_mov_b32 %0, v4\n\tv_mov_b32 %1, v5\n\t"
        "v_mov_b32 v0, %5\n\tv_mov_b32 v1, %5\n\tv_mov_b32 v2, %4\n\tv_mov_b32 v3, %4\n\t"
        "v_mov_b32 v4, %6\n\tv_mov_b32 v5, %6\n\t"
        "v_dual_fmac_f32 v4, v0, v2 :: v_dual_fmac_f32 v5, v1, v3\n\t"
        "v_mov_b32 %2, v4\n\tv_mov_b32 %3, v5"
        : "=&v"(xo), "=&v"(yo), "=&v"(xs), "=&v"(ys)
        : "v"(a), "v"(b), "v"(c)
        : "v0", "v1", "v2", "v3", "v4", "v5");
  }
}

// ---------------------------------------------------------------- integer oracle
struct Exp { uint32_t v, k; };

template <int OP>
__device__ __forceinline__ Exp oracle(uint32_t a, uint32_t b, uint32_t c, bool fi) {
  if (is_nan(a)) return Exp{a | 0x400000u, K_A};
  if (is_nan(b)) return Exp{b | 0x400000u, K_B};
  const bool az = is_zin(a, fi), bz = is_zin(b, fi);
  if (OP == OP_ADD) {
    if (is_inf(a) && is_inf(b) && ((a ^ b) >> 31)) return Exp{0xffc00000u, K_INVALID};
    return Exp{0, K_NONE};
  }
  const bool ia = is_inf(a), ib = is_inf(b);
  if ((ia && bz) || (ib && az)) return Exp{0xffc00000u, K_INVALID};
  if (OP == OP_MUL) return Exp{0, K_NONE};
  if (is_nan(c)) return Exp{c | 0x400000u, K_C};
  if (is_inf(c)) {
    if (ia || ib) {
      if (((a ^ b) >> 31) != (c >> 31)) return Exp{0xffc00000u, K_INFCONF};
      return Exp{0, K_NONE};
    }
    // finite product: may overflow to inf (rounding-mode dependent) -> unjudged when the exponent sum allows it
    if (!az && !bz && (((a >> 23) & 0xffu) + ((b >> 23) & 0xffu)) >= 380u) return Exp{0, K_UNKNOWN};
  }
  return Exp{0, K_NONE};
}
__device__ __forceinline__ uint32_t sel_bad(uint32_t r, Exp e) {
  if (e.k == K_A || e.k == K_B || e.k == K_C) return r != e.v;
  if (e.k == K_INVALID || e.k == K_INFCONF) return r != 0xffc00000u;
  return 0;
}

// ---------------------------------------------------------------- counters / samples
struct Cnt {
  uint32_t pk[8], tot[NC], n;
  __device__ __forceinline__ void init() {
#pragma unroll
    for (int j = 0; j < 8; ++j) pk[j] = 0;
#pragma unroll
    for (int k = 0; k < NC; ++k) tot[k] = 0;
    n = 0;
  }
  __device__ __forceinline__ void flush() {
#pragma unroll
    for (int k = 0; k < NC; ++k) tot[k] += (pk[k & 7] >> (8 * (k >> 3))) & 0xffu;
#pragma unroll
    for (int j = 0; j < 8; ++j) pk[j] = 0;
    n = 0;
  }
  // flag bit k -> byte (k>>3) of pk[k&7]; byte counters flushed every 255 adds
  __device__ __forceinline__ void add(uint32_t f) {
#pragma unroll
    for (int j = 0; j < 8; ++j) pk[j] += (f >> j) & 0x01010101u;
    if (++n == 255u) flush();
  }
};

struct Rec { uint32_t w[REC]; };

// sc: per-class attempt counters for this op (NCLS); samp: this op's records (NCLS*SLOTS*REC)
__device__ __noinline__ void sample_rec(uint32_t* sc, uint32_t* samp, uint32_t want, Rec r) {
  for (uint32_t cls = 0; cls < NCLS; ++cls) {
    if (!((want >> cls) & 1u)) continue;
    if (*(volatile uint32_t*)&sc[cls] >= SLOTS) continue;
    const uint32_t slot = atomicAdd(&sc[cls], 1u);
    if (slot >= SLOTS) continue;
    uint32_t* d = samp + ((size_t)cls * SLOTS + slot) * REC;
    for (int i = 0; i < REC; ++i) d[i] = r.w[i];
    d[14] = cls;
  }
}

struct Res { uint32_t no, ns, xo, yo, xs, ys; Exp eo, es; };

template <int OP, int DEN>
__device__ __forceinline__ void compute(uint32_t a, uint32_t b, uint32_t c, Res& r) {
  constexpr bool FI = !(DEN & 1);
  r.no = nat<OP>(a, b, c);
  r.ns = nat<OP>(b, a, c);
  dual<OP>(a, b, c, r.xo, r.yo, r.xs, r.ys);
  r.eo = oracle<OP>(a, b, c, FI);
  r.es = oracle<OP>(b, a, c, FI);
}

__device__ __forceinline__ Rec make_rec(uint32_t a, uint32_t b, uint32_t c, const Res& r, uint32_t safe, uint32_t op, uint32_t k0, uint32_t k1) {
  Rec s;
  s.w[0] = a; s.w[1] = b; s.w[2] = c; s.w[3] = r.eo.v; s.w[4] = r.es.v; s.w[5] = r.no; s.w[6] = r.ns;
  s.w[7] = r.xo; s.w[8] = r.yo; s.w[9] = r.xs; s.w[10] = r.ys;
  s.w[11] = r.eo.k | (r.es.k << 4) | (safe << 8);
  s.w[12] = k0; s.w[13] = k1; s.w[14] = 0; s.w[15] = op;
  return s;
}

template <int OP, int DEN>
__device__ __forceinline__ void eval(uint32_t a, uint32_t b, uint32_t c, Cnt& cn, uint32_t& tried, uint32_t* sc, uint32_t* samp,
                                     uint32_t k0, uint32_t k1) {
  Res r;
  compute<OP, DEN>(a, b, c, r);
  const uint32_t nanexp = (r.eo.k >= K_A && r.eo.k <= K_INFCONF);
  const uint32_t invexp = (r.eo.k == K_INVALID || r.eo.k == K_INFCONF);
  const uint32_t unk = (r.eo.k == K_UNKNOWN);
  const uint32_t s_no = sel_bad(r.no, r.eo), s_ns = sel_bad(r.ns, r.es), s_xo = sel_bad(r.xo, r.eo), s_yo = sel_bad(r.yo, r.eo),
                 s_xs = sel_bad(r.xs, r.es), s_ys = sel_bad(r.ys, r.es);
  const uint32_t s_any = s_no | s_ns | s_xo | s_yo | s_xs | s_ys;
  const uint32_t d_xo = r.xo != r.no, d_yo = r.yo != r.no, d_xs = r.xs != r.ns, d_ys = r.ys != r.ns;
  const uint32_t d_any = d_xo | d_yo | d_xs | d_ys;
  const uint32_t xy_o = r.xo != r.yo, xy_s = r.xs != r.ys;
  const uint32_t w_n = r.no != r.ns, w_x = r.xo != r.xs, w_y = r.yo != r.ys;
  const uint32_t w_any = w_n | w_x | w_y;
  const uint32_t both = is_nan(a) & is_nan(b);
  const uint32_t safe = (!both) | ((a | 0x400000u) == (b | 0x400000u));
  const uint32_t unexp = (r.eo.k == K_NONE) & (is_nan(r.no) | is_nan(r.ns) | is_nan(r.xo) | is_nan(r.yo) | is_nan(r.xs) | is_nan(r.ys));
  uint32_t f = 1u;
  f |= nanexp << C_NANEXP; f |= invexp << C_INVEXP; f |= unk << C_UNK;
  f |= s_no << C_SEL_NO; f |= s_ns << C_SEL_NS; f |= s_xo << C_SEL_XO; f |= s_yo << C_SEL_YO; f |= s_xs << C_SEL_XS; f |= s_ys << C_SEL_YS;
  f |= s_any << C_SEL_ANY;
  f |= d_xo << C_DU_XO; f |= d_yo << C_DU_YO; f |= d_xs << C_DU_XS; f |= d_ys << C_DU_YS; f |= d_any << C_DU_ANY;
  f |= xy_o << C_XY_O; f |= xy_s << C_XY_S;
  f |= w_n << C_SW_N; f |= w_x << C_SW_X; f |= w_y << C_SW_Y; f |= w_any << C_SW_ANY;
  f |= safe << C_SAFE; f |= (safe & w_any) << C_SAFE_BAD; f |= (safe ^ 1u) << C_UNSAFE; f |= ((safe ^ 1u) & w_any) << C_UNSAFE_SW;
  f |= both << C_BOTHNAN; f |= unexp << C_UNEXP;
  f |= (invexp & (r.no == 0x7fc00000u)) << C_INV7FC;
  f |= (invexp & (r.no == 0xffc00000u)) << C_INVFFC;
  f |= (invexp & (r.no != 0x7fc00000u) & (r.no != 0xffc00000u)) << C_INVOTH;
  f |= is_nan(r.no) << C_NANRES;
  cn.add(f);

  const uint32_t ok_sel = nanexp & (s_any ^ 1u);
  uint32_t want = 0;
  want |= s_any << CL_SELBAD;
  want |= d_any << CL_DUALBAD;
  want |= (safe & w_any) << CL_SAFEBAD;
  want |= ((safe ^ 1u) & w_any) << CL_SWAP;
  want |= (ok_sel & (r.eo.k == K_INVALID || r.eo.k == K_C || r.eo.k == K_INFCONF)) << CL_SELOK_C;
  want |= (ok_sel & (r.eo.k == K_A || r.eo.k == K_B) & (both ^ 1u)) << CL_SELOK_AB;
  want |= unexp << CL_UNEXP;
  want |= (both & safe) << CL_SAFE_BOTHNAN;
  want &= ~tried;
  if (want) {
    tried |= want;
    sample_rec(sc, samp, want, make_rec(a, b, c, r, safe, OP, k0, k1));
  }
}

template <int DEN>
__device__ __forceinline__ void eval3(uint32_t a, uint32_t b, uint32_t c, Cnt* cn, uint32_t* tried, uint32_t* sc, uint32_t* samp,
                                      uint32_t k0, uint32_t k1) {
  eval<OP_ADD, DEN>(a, b, c, cn[0], tried[0], sc + 0 * NCLS, samp + (size_t)0 * NCLS * SLOTS * REC, k0, k1);
  eval<OP_MUL, DEN>(a, b, c, cn[1], tried[1], sc + 1 * NCLS, samp + (size_t)1 * NCLS * SLOTS * REC, k0, k1);
  eval<OP_FMAC, DEN>(a, b, c, cn[2], tried[2], sc + 2 * NCLS, samp + (size_t)2 * NCLS * SLOTS * REC, k0, k1);
}

// ---------------------------------------------------------------- sweep kernel
struct KArgs {
  unsigned long long* cnt;  // NCNT u64
  uint32_t* sc;             // NOPS*NCLS
  uint32_t* samp;           // NOPS*NCLS*SLOTS*REC
  uint32_t* ext;            // 16
  uint32_t start, count;
  unsigned long long seed;
};

__device__ __forceinline__ uint32_t rgen(uint32_t x, uint32_t s) {
  switch (s) {
    case 4: return (x & 0x80000000u) | 0x7f800000u | nzm(x);
    case 5: return (x & 0x80000000u) | 0x7f800000u;
    case 6: return x & 0x807fffffu;
    case 7: return kV[(x >> 8) % 40u];
    default: return x;
  }
}

template <int M, int S>
__global__ __launch_bounds__(256) void sweep(KArgs k) {
  constexpr int DEN = M & 3;
  mode_prologue<M>(k.ext);
  Cnt cn[NOPS];
#pragma unroll
  for (int o = 0; o < NOPS; ++o) cn[o].init();
  uint32_t tried[NOPS] = {0, 0, 0};
  const uint32_t gid = blockIdx.x * 256u + threadIdx.x, stride = gridDim.x * 256u;
  const uint32_t end = k.start + k.count;
#pragma unroll 1
  for (uint32_t i = k.start + gid; i < end; i += stride) {
    if (S == 0) {
      const uint32_t sign = i >= 8388607u ? 1u : 0u;
      const uint32_t m = i - (sign ? 8388607u : 0u) + 1u;  // 1 .. 2^23-1
      const uint32_t a = (sign << 31) | 0x7f800000u | m;
#pragma unroll 1
      for (uint32_t p = 0; p < 16; ++p) {
        const uint32_t b = p < 12 ? kPart[p] : part_derived(p, a);
        const uint32_t ci = (m + p) & 7u;
        const uint32_t c = ci < 2 ? c_derived(ci, m) : kC[ci];
        eval3<DEN>(a, b, c, cn, tried, k.sc, k.samp, i, p);
      }
    } else if (S == 1) {
      const uint32_t a = kV[i / 1600u], b = kV[(i / 40u) % 40u], c = kV[i % 40u];
      eval3<DEN>(a, b, c, cn, tried, k.sc, k.samp, i, 0);
    } else {
      const unsigned long long h1 = mix64(k.seed ^ ((unsigned long long)i * 0x9e3779b97f4a7c15ull));
      const unsigned long long h2 = mix64(h1 + 0x1234567887654321ull);
      const uint32_t a = rgen((uint32_t)h1, (uint32_t)(h2 & 7u));
      const uint32_t b = rgen((uint32_t)(h1 >> 32), (uint32_t)((h2 >> 3) & 7u));
      const uint32_t c = rgen((uint32_t)(h2 >> 32), (uint32_t)((h2 >> 6) & 7u));
      eval3<DEN>(a, b, c, cn, tried, k.sc, k.samp, i, 0);
    }
  }
#pragma unroll
  for (int o = 0; o < NOPS; ++o) cn[o].flush();
  __shared__ uint32_t sh[8][NCNT];
  const uint32_t w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
  for (int o = 0; o < NOPS; ++o) {
#pragma unroll
    for (int j = 0; j < NC; ++j) {
      uint32_t v = cn[o].tot[j];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
      if (l == 0) sh[w][o * NC + j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x < NCNT) {
    uint32_t s = 0;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) s += sh[ww][threadIdx.x];
    if (s) atomicAdd(&k.cnt[threadIdx.x], (unsigned long long)s);
  }
}

// ---------------------------------------------------------------- edge-row kernel: raw rows (16 u32) per (triple, op)
template <int OP, int DEN>
__device__ __forceinline__ void edge_fill(uint32_t a, uint32_t b, uint32_t c, uint32_t* row, uint32_t t) {
  Res r;
  compute<OP, DEN>(a, b, c, r);
  const uint32_t both = is_nan(a) & is_nan(b);
  const uint32_t safe = (!both) | ((a | 0x400000u) == (b | 0x400000u));
  const Rec s = make_rec(a, b, c, r, safe, OP, t, 0);
  for (int i = 0; i < REC; ++i) row[i] = s.w[i];
}

template <int M>
__global__ __launch_bounds__(256) void edgek(const uint32_t* tri, uint32_t* rows, uint32_t n, uint32_t* ext) {
  constexpr int DEN = M & 3;
  mode_prologue<M>(ext);
  const uint32_t i = blockIdx.x * 256u + threadIdx.x;
  if (i < n * 3u) {
    const uint32_t t = i / 3u, op = i - t * 3u;
    const uint32_t a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    uint32_t* row = rows + (size_t)i * REC;
    if (op == 0) edge_fill<OP_ADD, DEN>(a, b, c, row, t);
    else if (op == 1) edge_fill<OP_MUL, DEN>(a, b, c, row, t);
    else edge_fill<OP_FMAC, DEN>(a, b, c, row, t);
  }
}

// ---------------------------------------------------------------- host
template <int M>
static void lm(int s, const KArgs& k, int blocks) {
  if (s == 0) sweep<M, 0><<<blocks, 256>>>(k);
  else if (s == 1) sweep<M, 1><<<blocks, 256>>>(k);
  else sweep<M, 2><<<blocks, 256>>>(k);
}
template <int M>
static void le(const uint32_t* tri, uint32_t* rows, uint32_t n, uint32_t* ext) {
  edgek<M><<<(n * 3 + 255) / 256, 256>>>(tri, rows, n, ext);
}
typedef void (*LmFn)(int, const KArgs&, int);
typedef void (*LeFn)(const uint32_t*, uint32_t*, uint32_t, uint32_t*);
static const LmFn kLm[16] = {lm<0>, lm<1>, lm<2>, lm<3>, lm<4>, lm<5>, lm<6>, lm<7>, lm<8>, lm<9>, lm<10>, lm<11>, lm<12>, lm<13>, lm<14>, lm<15>};
static const LeFn kLe[16] = {le<0>, le<1>, le<2>, le<3>, le<4>, le<5>, le<6>, le<7>, le<8>, le<9>, le<10>, le<11>, le<12>, le<13>, le<14>, le<15>};

static const char* kCN[NC] = {
    "cases", "nan_expected", "invalid_expected", "unknown_unchecked", "sel_native_orig", "sel_native_swap", "sel_dual_x_orig",
    "sel_dual_y_orig", "sel_dual_x_swap", "sel_dual_y_swap", "sel_any", "dual_vs_native_x_orig", "dual_vs_native_y_orig",
    "dual_vs_native_x_swap", "dual_vs_native_y_swap", "dual_vs_native_any", "x_vs_y_orig", "x_vs_y_swap",
    "swap_unequal_native", "swap_unequal_dual_x", "swap_unequal_dual_y", "swap_unequal_any", "safe_cases", "safe_swap_mismatch",
    "unsafe_cases", "unsafe_swap_unequal", "both_nan_cases", "unexpected_nan", "invalid_result_7fc00000", "invalid_result_ffc00000",
    "invalid_result_other", "native_orig_result_nan"};
static const char* kClsN[NCLS] = {"sel_mismatch", "dual_mismatch", "safe_mismatch", "swap_unequal_unsafe", "sel_ok_invalid_or_c",
                                  "sel_ok_single_nan", "unexpected_nan", "safe_both_nan"};
static const char* kOpN[NOPS] = {"add", "mul", "fmac"};
static const char* kSuiteN[4] = {"nan", "tri", "rand", "edges"};

struct Cell {
  bool done = false, skipped = false;
  unsigned long long cnt[NCNT] = {0};
  uint32_t ext[16] = {0};
  uint32_t sc[NOPS * NCLS] = {0};
  std::vector<uint32_t> samp;
  double secs = 0;
  int launches = 0;
  unsigned long long expect = 0;  // expected cases per op
};

static Cell g_cell[16][3];
static uint32_t g_edge_ext[16][16];
static bool g_edge_done[16];
static std::vector<uint32_t> g_edge_rows[16];
static std::vector<uint32_t> g_edge_tri;
static uint32_t g_n_edge = 0;

static std::string g_out;
static unsigned long long g_seed = 0x5eed0bad1ee70001ull, g_nrand = 4194304ull;
static uint32_t g_chunk = 2097152u;
static double g_budget = 0;
static std::string g_arch, g_dev, g_uuid, g_uuid_raw, g_uuid_rocr, g_uuid_src, g_host, g_argv, g_suites_arg;
static int g_pci_dom = 0, g_pci_bus = 0, g_pci_dev = 0;
static int g_modes[16], g_nmodes = 0;
static bool g_sel[4] = {false, false, false, false};
static bool g_complete = false;
static double g_total_secs = 0;

static FILE* openw(const char* name) {
  const std::string p = g_out + "/" + name;
  FILE* f = fopen(p.c_str(), "w");
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
  return f;
}
static void closew(FILE* f) { fflush(f); fsync(fileno(f)); fclose(f); }

static void jstr(FILE* f, const char* s) {
  fputc('"', f);
  for (; *s; ++s) {
    if (*s == '"' || *s == '\\') { fputc('\\', f); fputc(*s, f); }
    else if ((unsigned char)*s < 0x20) fprintf(f, "\\u%04x", (unsigned)*s);
    else fputc(*s, f);
  }
  fputc('"', f);
}

static void build_edges() {
  const uint32_t ea[12] = {0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xff800001u, 0x7fc00001u, 0xffc00001u,
                           0x7fbfffffu, 0xffbfffffu, 0x7fffffffu, 0xffffffffu, 0x7fc0beefu, 0xffa0beefu};
  for (int i = 0; i < 12; ++i)
    for (uint32_t p = 0; p < 16; ++p) {
      const uint32_t a = ea[i], m = a & 0x7fffffu;
      const uint32_t b = p < 12 ? H_PART[p] : part_derived(p, a);
      const uint32_t ci = (m + p) & 7u;
      const uint32_t c = ci < 2 ? c_derived(ci, m) : H_C[ci];
      g_edge_tri.push_back(a); g_edge_tri.push_back(b); g_edge_tri.push_back(c);
    }
  const uint32_t ia[2] = {0x7f800000u, 0xff800000u};
  const uint32_t ib[8] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu, 0x3f800000u, 0xbf800000u};
  const uint32_t ic[5] = {0x7fc00000u, 0xffc00000u, 0x7f800000u, 0xff800000u, 0x3f800000u};
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 8; ++j) for (int k = 0; k < 5; ++k) {
    g_edge_tri.push_back(ia[i]); g_edge_tri.push_back(ib[j]); g_edge_tri.push_back(ic[k]);
  }
  const uint32_t ex[7][3] = {{0x7f800000u, 0xff800000u, 0x3f800000u}, {0xff800000u, 0x7f800000u, 0x3f800000u},
                             {0x7f800000u, 0x7f800000u, 0xff800000u}, {0x7f800000u, 0x7f800000u, 0x7f800000u},
                             {0x7f7fffffu, 0x7f7fffffu, 0xff800000u}, {0x7f7fffffu, 0xff7fffffu, 0x7f800000u},
                             {0x7f7fffffu, 0x3f800000u, 0xff800000u}};
  for (int i = 0; i < 7; ++i) for (int j = 0; j < 3; ++j) g_edge_tri.push_back(ex[i][j]);
  g_n_edge = (uint32_t)(g_edge_tri.size() / 3);
}

static unsigned long long expected_cases(int suite) {
  return suite == 0 ? (unsigned long long)NNAN_ITEMS * 16ull : suite == 1 ? (unsigned long long)NTRI : g_nrand;
}

static void write_all() {
  // ---- summary.tsv
  FILE* f = openw("nan_swap_summary.tsv");
  fprintf(f, "mode\tdenorm\tround\tsuite\top\texpected_cases");
  for (int k = 0; k < NC; ++k) fprintf(f, "\t%s", kCN[k]);
  fprintf(f, "\toutputs\tcount_ok\n");
  for (int mi = 0; mi < g_nmodes; ++mi) {
    const int m = g_modes[mi];
    for (int s = 0; s < 3; ++s) {
      const Cell& c = g_cell[m][s];
      if (!c.done) continue;
      for (int o = 0; o < NOPS; ++o) {
        fprintf(f, "%d\t%d\t%d\t%s\t%s\t%llu", m, m & 3, m >> 2, kSuiteN[s], kOpN[o], c.expect);
        for (int k = 0; k < NC; ++k) fprintf(f, "\t%llu", c.cnt[o * NC + k]);
        fprintf(f, "\t%llu\t%d\n", c.cnt[o * NC + C_CASES] * 6ull, c.cnt[o * NC + C_CASES] == c.expect ? 1 : 0);
      }
    }
  }
  closew(f);
  // ---- modes.tsv
  f = openw("nan_swap_modes.tsv");
  fprintf(f, "mode\tdenorm\tround\tsuite\twaves\tctl_bad_waves\tmode_bad_waves\tmode_readback_or\tmode_readback_and\tctl_in\tctl_out\tctl_round_up\tctl_round_dn\texp_ctl_in\texp_ctl_out\texp_ctl_round_up\texp_ctl_round_dn\tsecs\tlaunches\n");
  for (int mi = 0; mi < g_nmodes; ++mi) {
    const int m = g_modes[mi];
    for (int s = 0; s < 4; ++s) {
      const uint32_t* e = s < 3 ? g_cell[m][s].ext : g_edge_ext[m];
      const bool done = s < 3 ? g_cell[m][s].done : g_edge_done[m];
      if (!done) continue;
      fprintf(f, "%d\t%d\t%d\t%s\t%u\t%u\t%u\t%08x\t%08x\t%08x\t%08x\t%08x\t%08x\t%08x\t%08x\t%08x\t%08x\t%.3f\t%d\n", m, m & 3, m >> 2, kSuiteN[s],
              e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], e[12], s < 3 ? g_cell[m][s].secs : 0.0,
              s < 3 ? g_cell[m][s].launches : 1);
    }
  }
  closew(f);
  // ---- samples.tsv (sorted by key within (suite,mode,op,class))
  f = openw("nan_swap_samples.tsv");
  fprintf(f, "suite\tmode\top\tclass\tslot\ta\tb\tc\texp_orig\texp_swap\tnative_orig\tnative_swap\tdualx_orig\tdualy_orig\tdualx_swap\tdualy_swap\tkind_orig\tkind_swap\tsafe\tkey0\tkey1\tattempts\n");
  for (int mi = 0; mi < g_nmodes; ++mi) {
    const int m = g_modes[mi];
    for (int s = 0; s < 3; ++s) {
      const Cell& c = g_cell[m][s];
      if (!c.done) continue;
      for (int o = 0; o < NOPS; ++o)
        for (int cl = 0; cl < NCLS; ++cl) {
          const uint32_t att = c.sc[o * NCLS + cl], n = att < SLOTS ? att : SLOTS;
          std::vector<const uint32_t*> v;
          for (uint32_t i = 0; i < n; ++i) v.push_back(&c.samp[(((size_t)o * NCLS + cl) * SLOTS + i) * REC]);
          std::sort(v.begin(), v.end(), [](const uint32_t* x, const uint32_t* y) { return x[12] != y[12] ? x[12] < y[12] : x[13] < y[13]; });
          for (uint32_t i = 0; i < n; ++i) {
            const uint32_t* w = v[i];
            fprintf(f, "%s\t%d\t%s\t%s\t%u", kSuiteN[s], m, kOpN[o], kClsN[cl], i);
            for (int j = 0; j < 11; ++j) fprintf(f, "\t%08x", w[j]);
            fprintf(f, "\t%u\t%u\t%u\t%u\t%u\t%u\n", w[11] & 15u, (w[11] >> 4) & 15u, (w[11] >> 8) & 1u, w[12], w[13], att);
          }
        }
    }
  }
  closew(f);
  // ---- edges.tsv
  f = openw("nan_swap_edges.tsv");
  fprintf(f, "mode\top\ta\tb\tc\texp_orig\texp_swap\tnative_orig\tnative_swap\tdualx_orig\tdualy_orig\tdualx_swap\tdualy_swap\tkind_orig\tkind_swap\tsafe\ttriple_idx\n");
  for (int mi = 0; mi < g_nmodes; ++mi) {
    const int m = g_modes[mi];
    if (!g_edge_done[m]) continue;
    for (uint32_t i = 0; i < g_n_edge * 3u; ++i) {
      const uint32_t* w = &g_edge_rows[m][(size_t)i * REC];
      fprintf(f, "%d\t%s", m, kOpN[w[15]]);
      for (int j = 0; j < 11; ++j) fprintf(f, "\t%08x", w[j]);
      fprintf(f, "\t%u\t%u\t%u\t%u\n", w[11] & 15u, (w[11] >> 4) & 15u, (w[11] >> 8) & 1u, w[12]);
    }
  }
  closew(f);
  // ---- summary.json
  unsigned long long tsel = 0, tdu = 0, tsafe = 0, tunexp = 0, tcases = 0, tout = 0, tswap = 0, tctl = 0, tmode = 0, tcountbad = 0;
  for (int m = 0; m < 16; ++m) {
    for (int s = 0; s < 3; ++s) {
      const Cell& c = g_cell[m][s];
      if (!c.done) continue;
      tctl += c.ext[1]; tmode += c.ext[2];
      for (int o = 0; o < NOPS; ++o) {
        tsel += c.cnt[o * NC + C_SEL_ANY]; tdu += c.cnt[o * NC + C_DU_ANY]; tsafe += c.cnt[o * NC + C_SAFE_BAD];
        tunexp += c.cnt[o * NC + C_UNEXP]; tcases += c.cnt[o * NC + C_CASES]; tswap += c.cnt[o * NC + C_SW_ANY];
        if (c.cnt[o * NC + C_CASES] != c.expect) ++tcountbad;
      }
    }
    if (g_edge_done[m]) { tctl += g_edge_ext[m][1]; tmode += g_edge_ext[m][2]; }
  }
  tout = tcases * 6ull;
  f = openw("nan_swap_summary.json");
  fprintf(f, "{\n \"probe\":\"nan_swap\",\"complete\":%s,\"arch\":", g_complete ? "true" : "false");
  jstr(f, g_arch.c_str());
  fprintf(f, ",\"device\":"); jstr(f, g_dev.c_str());
  fprintf(f, ",\"uuid\":"); jstr(f, g_uuid.c_str());
  fprintf(f, ",\"uuid_source\":"); jstr(f, g_uuid_src.c_str());
  fprintf(f, ",\"uuid_raw_bytes_hex\":"); jstr(f, g_uuid_raw.c_str());
  fprintf(f, ",\"uuid_rocr_visible_devices\":"); jstr(f, g_uuid_rocr.c_str());
  fprintf(f, ",\"uuid_matches_rocr_visible_devices\":%s", (!g_uuid_rocr.empty() && g_uuid == g_uuid_rocr) ? "true" : "false");
  fprintf(f, ",\"pci\":\"%04x:%02x:%02x\",\"host\":", g_pci_dom, g_pci_bus, g_pci_dev);
  jstr(f, g_host.c_str());
  fprintf(f, ",\"argv\":"); jstr(f, g_argv.c_str());
  fprintf(f, ",\n \"seed\":\"0x%016llx\",\"random_triples_per_mode_op\":%llu,\"nan_items\":%u,\"nan_partners\":16,\"tri_triples\":%u,\"chunk_items\":%u,\"budget_secs\":%.1f,\"total_secs\":%.3f,\n",
          g_seed, g_nrand, NNAN_ITEMS, NTRI, g_chunk, g_budget, g_total_secs);
  fprintf(f, " \"suites_arg\":"); jstr(f, g_suites_arg.c_str());
  fprintf(f, ",\"modes_run\":[");
  for (int i = 0; i < g_nmodes; ++i) fprintf(f, "%s%d", i ? "," : "", g_modes[i]);
  fprintf(f, "],\n \"mode_definition\":\"mode m: FP_DENORM single (MODE[5:4]) = m&3, FP_ROUND single (MODE[1:0]) = m>>2; no IEEE/DX10_CLAMP bits on gfx12, reserved bits untouched\",\n");
  fprintf(f, " \"env\":{");
  bool first = true;
  static const char* pre[] = {"FABRIC", "HIP_", "ROCR_", "HSA_", "NP_", "NS_", "CARD", "GPU_", "CUDA_VISIBLE", "TMPDIR", "LD_LIBRARY_PATH"};
  for (char** e = environ; e && *e; ++e) {
    bool take = false;
    for (const char* p : pre) if (!strncmp(*e, p, strlen(p))) take = true;
    if (!take) continue;
    const char* eq = strchr(*e, '=');
    if (!eq) continue;
    std::string k(*e, eq - *e);
    fprintf(f, "%s", first ? "" : ",");
    first = false;
    jstr(f, k.c_str()); fputc(':', f); jstr(f, eq + 1);
  }
  fprintf(f, "},\n \"hash_note\":\"source and binary sha256 are computed externally by the parent (build json)\",\n");
  fprintf(f, " \"totals\":{\"cases\":%llu,\"outputs\":%llu,\"selection_mismatch_cases\":%llu,\"native_dual_mismatch_cases\":%llu,\"safe_criterion_mismatch_cases\":%llu,\"unexpected_nan_cases\":%llu,\"swap_unequal_cases_informational\":%llu,\"control_bad_waves\":%llu,\"mode_readback_bad_waves\":%llu,\"case_count_mismatch_cells\":%llu},\n",
          tcases, tout, tsel, tdu, tsafe, tunexp, tswap, tctl, tmode, tcountbad);
  fprintf(f, " \"cells\":[\n");
  bool fc = true;
  for (int mi = 0; mi < g_nmodes; ++mi) {
    const int m = g_modes[mi];
    for (int s = 0; s < 3; ++s) {
      const Cell& c = g_cell[m][s];
      if (!c.done && !c.skipped) continue;
      fprintf(f, "%s  {\"mode\":%d,\"denorm\":%d,\"round\":%d,\"suite\":\"%s\",\"done\":%s,\"secs\":%.3f,\"launches\":%d,\"expected_cases_per_op\":%llu,",
              fc ? "" : ",\n", m, m & 3, m >> 2, kSuiteN[s], c.done ? "true" : "false", c.secs, c.launches, c.expect);
      fc = false;
      if (!c.done) { fprintf(f, "\"skipped\":true}"); continue; }
      fprintf(f, "\"control\":{\"waves\":%u,\"ctl_bad_waves\":%u,\"mode_bad_waves\":%u,\"mode_or\":\"%08x\",\"mode_and\":\"%08x\",\"obs\":[\"%08x\",\"%08x\",\"%08x\",\"%08x\"],\"exp\":[\"%08x\",\"%08x\",\"%08x\",\"%08x\"]},\"ops\":{",
              c.ext[0], c.ext[1], c.ext[2], c.ext[3], c.ext[4], c.ext[5], c.ext[6], c.ext[7], c.ext[8], c.ext[9], c.ext[10], c.ext[11], c.ext[12]);
      for (int o = 0; o < NOPS; ++o) {
        fprintf(f, "%s\"%s\":{", o ? "," : "", kOpN[o]);
        for (int k = 0; k < NC; ++k) fprintf(f, "%s\"%s\":%llu", k ? "," : "", kCN[k], c.cnt[o * NC + k]);
        fprintf(f, ",\"outputs\":%llu,\"count_ok\":%s}", c.cnt[o * NC + C_CASES] * 6ull, c.cnt[o * NC + C_CASES] == c.expect ? "true" : "false");
      }
      fprintf(f, "}}");
    }
  }
  fprintf(f, "\n ],\n \"edges\":{\"triples\":%u,\"rows_per_mode\":%u,\"file\":\"nan_swap_edges.tsv\"},\n", g_n_edge, g_n_edge * 3u);
  // first <=32 discrepant raw tuples (selection / dual / safe / unexpected-NaN classes)
  fprintf(f, " \"first_discrepant_raw\":[");
  int nd = 0;
  for (int mi = 0; mi < g_nmodes && nd < 32; ++mi)
    for (int s = 0; s < 3 && nd < 32; ++s) {
      const Cell& c = g_cell[g_modes[mi]][s];
      if (!c.done) continue;
      for (int o = 0; o < NOPS && nd < 32; ++o)
        for (int cl = 0; cl < NCLS && nd < 32; ++cl) {
          if (cl != CL_SELBAD && cl != CL_DUALBAD && cl != CL_SAFEBAD && cl != CL_UNEXP) continue;
          const uint32_t att = c.sc[o * NCLS + cl], n = att < SLOTS ? att : SLOTS;
          for (uint32_t i = 0; i < n && nd < 32; ++i) {
            const uint32_t* w = &c.samp[(((size_t)o * NCLS + cl) * SLOTS + i) * REC];
            fprintf(f, "%s\n  {\"mode\":%d,\"suite\":\"%s\",\"op\":\"%s\",\"class\":\"%s\",\"raw\":[", nd ? "," : "", g_modes[mi], kSuiteN[s], kOpN[o], kClsN[cl]);
            for (int j = 0; j < 11; ++j) fprintf(f, "%s\"%08x\"", j ? "," : "", w[j]);
            fprintf(f, "],\"key\":[%u,%u]}", w[12], w[13]);
            ++nd;
          }
        }
    }
  fprintf(f, "\n ],\n \"samples_file\":\"nan_swap_samples.tsv\",\"summary_tsv\":\"nan_swap_summary.tsv\",\"modes_tsv\":\"nan_swap_modes.tsv\"\n}\n");
  closew(f);
}

static void parse_modes(const char* s) {
  bool use[16] = {false};
  std::string t = s;
  size_t p = 0;
  while (p <= t.size()) {
    size_t q = t.find(',', p);
    if (q == std::string::npos) q = t.size();
    std::string tok = t.substr(p, q - p);
    if (!tok.empty()) {
      int lo, hi;
      if (sscanf(tok.c_str(), "%d-%d", &lo, &hi) == 2) {}
      else if (sscanf(tok.c_str(), "%d", &lo) == 1) hi = lo;
      else { fprintf(stderr, "bad NS_MODES token %s\n", tok.c_str()); exit(2); }
      if (lo < 0 || hi > 15 || lo > hi) { fprintf(stderr, "bad NS_MODES range %s\n", tok.c_str()); exit(2); }
      for (int i = lo; i <= hi; ++i) use[i] = true;
    }
    p = q + 1;
  }
  g_nmodes = 0;
  for (int i = 0; i < 16; ++i) if (use[i]) g_modes[g_nmodes++] = i;
  if (!g_nmodes) { fprintf(stderr, "NS_MODES empty\n"); exit(2); }
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s OUTDIR [all|nan,tri,rand,edges]\n", argv[0]); return 2; }
  g_out = argv[1];
  mkdir(g_out.c_str(), 0755);
  if (access(g_out.c_str(), W_OK)) { fprintf(stderr, "OUTDIR %s not writable\n", g_out.c_str()); return 2; }
  for (int i = 0; i < argc; ++i) { if (i) g_argv += ' '; g_argv += argv[i]; }
  g_suites_arg = argc > 2 ? argv[2] : "all";
  {
    std::string s = "," + g_suites_arg + ",";
    const bool all = s.find(",all,") != std::string::npos;
    for (int i = 0; i < 4; ++i) g_sel[i] = all || s.find(std::string(",") + kSuiteN[i] + ",") != std::string::npos;
    if (!g_sel[0] && !g_sel[1] && !g_sel[2] && !g_sel[3]) { fprintf(stderr, "no suite selected by '%s'\n", g_suites_arg.c_str()); return 2; }
  }
  if (const char* e = getenv("NS_SEED")) g_seed = strtoull(e, nullptr, 0);
  if (const char* e = getenv("NS_RANDOM")) g_nrand = strtoull(e, nullptr, 0);
  if (const char* e = getenv("NS_CHUNK")) g_chunk = (uint32_t)strtoul(e, nullptr, 0);
  if (const char* e = getenv("NS_BUDGET_SECS")) g_budget = atof(e);
  if (g_nrand < 1048576ull || g_nrand > 0x7fffffffull) { fprintf(stderr, "NS_RANDOM must be in [1048576, 2^31)\n"); return 2; }
  if (g_chunk == 0) { fprintf(stderr, "NS_CHUNK must be nonzero\n"); return 2; }
  parse_modes(getenv("NS_MODES") ? getenv("NS_MODES") : "0-15");
  {
    char hn[256] = {0};
    gethostname(hn, sizeof hn - 1);
    g_host = hn;
  }

  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0));
  g_arch = p.gcnArchName; g_dev = p.name;
  g_pci_dom = p.pciDomainID; g_pci_bus = p.pciBusID; g_pci_dev = p.pciDeviceID;
  {
    // HIP reports the 16 UUID bytes as the ASCII hex digits of the ROCr unique id (GPU-<16 hex chars>), so label it by
    // normalising the ASCII; any other content is reported as raw hex under a RAW- label (never as a fake GPU- id).
    char raw[33];
    bool ascii = true;
    for (int i = 0; i < 16; ++i) {
      const unsigned char ch = (unsigned char)p.uuid.bytes[i];
      snprintf(raw + 2 * i, 3, "%02x", (unsigned)ch);
      if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F'))) ascii = false;
    }
    g_uuid_raw = raw;
    g_uuid = ascii ? std::string("GPU-") + std::string(p.uuid.bytes, 16) : std::string("RAW-") + raw;
    g_uuid_src = ascii ? "hip_ascii_hex" : "hip_raw_bytes";
    const char* rv = getenv("ROCR_VISIBLE_DEVICES");
    g_uuid_rocr = rv ? rv : "";
  }
  printf("device: %s %s uuid=%s pci=%04x:%02x:%02x warp=%d CUs=%d\n", p.name, p.gcnArchName, g_uuid.c_str(), g_pci_dom, g_pci_bus, g_pci_dev, p.warpSize, p.multiProcessorCount);
  printf("seed=0x%016llx random_triples=%llu nan_items=%u (x16 partners) tri=%u modes=%d suites=%s budget=%.1fs\n", g_seed, g_nrand, NNAN_ITEMS, NTRI, g_nmodes,
         g_suites_arg.c_str(), g_budget);
  fflush(stdout);
  if (strncmp(p.gcnArchName, "gfx1201", 7) != 0) { fprintf(stderr, "FATAL: this probe is gfx1201 only, device is %s\n", p.gcnArchName); return 5; }
  if (p.warpSize != 32) { fprintf(stderr, "FATAL: wave32 required, warpSize=%d\n", p.warpSize); return 5; }

  build_edges();
  const int blocks = 2048;
  unsigned long long* d_cnt; uint32_t *d_sc, *d_samp, *d_ext, *d_tri, *d_rows;
  const size_t cnt_b = NCNT * sizeof(unsigned long long), sc_b = NOPS * NCLS * 4, samp_b = (size_t)NOPS * NCLS * SLOTS * REC * 4;
  CK(hipMalloc(&d_cnt, cnt_b)); CK(hipMalloc(&d_sc, sc_b)); CK(hipMalloc(&d_samp, samp_b)); CK(hipMalloc(&d_ext, 16 * 4));
  CK(hipMalloc(&d_tri, g_edge_tri.size() * 4)); CK(hipMalloc(&d_rows, (size_t)g_n_edge * 3 * REC * 4));
  CK(hipMemcpy(d_tri, g_edge_tri.data(), g_edge_tri.size() * 4, hipMemcpyHostToDevice));

  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  bool budget_hit = false;
  auto reset_dev = [&]() {
    CK(hipMemset(d_cnt, 0, cnt_b)); CK(hipMemset(d_sc, 0, sc_b)); CK(hipMemset(d_samp, 0, samp_b));
    uint32_t e[16] = {0}; e[4] = 0xffffffffu;
    CK(hipMemcpy(d_ext, e, 64, hipMemcpyHostToDevice));
  };
  for (int mi = 0; mi < g_nmodes; ++mi) {
    const int m = g_modes[mi];
    for (int s = 0; s < 3; ++s) {
      if (!g_sel[s]) continue;
      Cell& c = g_cell[m][s];
      c.expect = expected_cases(s);
      if (g_budget > 0 && elapsed() > g_budget) { c.skipped = true; budget_hit = true; continue; }
      reset_dev();
      const unsigned long long total = s == 0 ? NNAN_ITEMS : s == 1 ? NTRI : g_nrand;
      const unsigned long long chunk = s == 0 ? g_chunk : total;
      const double ts = elapsed();
      for (unsigned long long st = 0; st < total; st += chunk) {
        KArgs ka;
        ka.cnt = d_cnt; ka.sc = d_sc; ka.samp = d_samp; ka.ext = d_ext;
        ka.start = (uint32_t)st; ka.count = (uint32_t)(std::min<unsigned long long>(chunk, total - st)); ka.seed = g_seed;
        kLm[m](s, ka, blocks);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        ++c.launches;
      }
      c.secs = elapsed() - ts;
      CK(hipMemcpy(c.cnt, d_cnt, cnt_b, hipMemcpyDeviceToHost));
      CK(hipMemcpy(c.sc, d_sc, sc_b, hipMemcpyDeviceToHost));
      c.samp.resize((size_t)NOPS * NCLS * SLOTS * REC);
      CK(hipMemcpy(c.samp.data(), d_samp, samp_b, hipMemcpyDeviceToHost));
      CK(hipMemcpy(c.ext, d_ext, 64, hipMemcpyDeviceToHost));
      c.done = true;
      printf("mode %2d (den %d rnd %d) %-4s %7.2fs launches=%d | cases/op=%llu(exp %llu) sel=%llu dual=%llu safe_bad=%llu swap_unequal=%llu unexp_nan=%llu ctl_bad=%u mode_bad=%u\n", m, m & 3, m >> 2, kSuiteN[s], c.secs,
             c.launches, c.cnt[C_CASES], c.expect, c.cnt[C_SEL_ANY] + c.cnt[NC + C_SEL_ANY] + c.cnt[2 * NC + C_SEL_ANY],
             c.cnt[C_DU_ANY] + c.cnt[NC + C_DU_ANY] + c.cnt[2 * NC + C_DU_ANY], c.cnt[C_SAFE_BAD] + c.cnt[NC + C_SAFE_BAD] + c.cnt[2 * NC + C_SAFE_BAD],
             c.cnt[C_SW_ANY] + c.cnt[NC + C_SW_ANY] + c.cnt[2 * NC + C_SW_ANY], c.cnt[C_UNEXP] + c.cnt[NC + C_UNEXP] + c.cnt[2 * NC + C_UNEXP], c.ext[1], c.ext[2]);
      fflush(stdout);
    }
    if (g_sel[3]) {
      reset_dev();
      kLe[m](d_tri, d_rows, g_n_edge, d_ext);
      CK(hipGetLastError());
      CK(hipDeviceSynchronize());
      g_edge_rows[m].resize((size_t)g_n_edge * 3 * REC);
      CK(hipMemcpy(g_edge_rows[m].data(), d_rows, g_edge_rows[m].size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(g_edge_ext[m], d_ext, 64, hipMemcpyDeviceToHost));
      g_edge_done[m] = true;
      printf("mode %2d edges rows=%u ctl_bad=%u mode_bad=%u\n", m, g_n_edge * 3u, g_edge_ext[m][1], g_edge_ext[m][2]);
      fflush(stdout);
    }
    g_total_secs = elapsed();
    write_all();  // per-mode flush: a killed fabric hold still leaves complete summaries for every finished mode
  }
  g_total_secs = elapsed();
  g_complete = !budget_hit;
  write_all();

  // totals + discrepant tuples on stdout
  unsigned long long tsel = 0, tdu = 0, tsafe = 0, tunexp = 0, tcases = 0, tswap = 0, tctl = 0, tmode = 0, tcountbad = 0;
  for (int mi = 0; mi < g_nmodes; ++mi) {
    const int m = g_modes[mi];
    for (int s = 0; s < 3; ++s) {
      const Cell& c = g_cell[m][s];
      if (!c.done) continue;
      tctl += c.ext[1]; tmode += c.ext[2];
      for (int o = 0; o < NOPS; ++o) {
        tsel += c.cnt[o * NC + C_SEL_ANY]; tdu += c.cnt[o * NC + C_DU_ANY]; tsafe += c.cnt[o * NC + C_SAFE_BAD];
        tcases += c.cnt[o * NC + C_CASES]; tswap += c.cnt[o * NC + C_SW_ANY]; tunexp += c.cnt[o * NC + C_UNEXP];
        if (c.cnt[o * NC + C_CASES] != c.expect) ++tcountbad;
      }
    }
    if (g_edge_done[m]) { tctl += g_edge_ext[m][1]; tmode += g_edge_ext[m][2]; }
  }
  int nd = 0;
  for (int mi = 0; mi < g_nmodes && nd < 32; ++mi)
    for (int s = 0; s < 3 && nd < 32; ++s) {
      const Cell& c = g_cell[g_modes[mi]][s];
      if (!c.done) continue;
      for (int o = 0; o < NOPS && nd < 32; ++o)
        for (int cl = 0; cl < NCLS && nd < 32; ++cl) {
          if (cl != CL_SELBAD && cl != CL_DUALBAD && cl != CL_SAFEBAD && cl != CL_UNEXP) continue;
          const uint32_t att = c.sc[o * NCLS + cl], n = att < SLOTS ? att : SLOTS;
          for (uint32_t i = 0; i < n && nd < 32; ++i) {
            const uint32_t* w = &c.samp[(((size_t)o * NCLS + cl) * SLOTS + i) * REC];
            printf("DISCREPANT %s mode=%d suite=%s op=%s a=%08x b=%08x c=%08x exp=%08x/%08x nat=%08x/%08x dualX=%08x/%08x dualY=%08x/%08x\n", kClsN[cl], g_modes[mi],
                   kSuiteN[s], kOpN[o], w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[9], w[8], w[10]);
            ++nd;
          }
        }
    }
  printf("TOTAL cases=%llu outputs=%llu selection_mismatch=%llu native_dual_mismatch=%llu safe_criterion_mismatch=%llu unexpected_nan=%llu swap_unequal(info)=%llu ctl_bad_waves=%llu mode_bad_waves=%llu case_count_bad_cells=%llu complete=%d secs=%.2f\n",
         tcases, tcases * 6ull, tsel, tdu, tsafe, tunexp, tswap, tctl, tmode, tcountbad, g_complete ? 1 : 0, g_total_secs);
  fflush(stdout);
  if (tsel || tdu || tsafe || tunexp) return 1;
  if (tctl || tmode || tcountbad) return 6;
  if (!g_complete) return 4;
  return 0;
}
