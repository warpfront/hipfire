// numerics_probe.hip -- raw hardware numerics capture for the PM ISA fill (gfx1151 / gfx1201 only).
//
// Opcodes (each issued by an explicit inline-asm mnemonic, never a compiler builtin):
//   v_dot2_f32_f16 (0x13)  v_dot2_f32_bf16 (0x1a)  v_dot4_i32_iu8 (0x16)  v_dot8_i32_iu4 (0x18)   [gfx11 + gfx12]
//   v_wmma_f32_16x16x16_bf16 (0x41)  v_wmma_bf16_16x16x16_bf16 (0x43)  v_wmma_f16_16x16x16_f16 (0x42)
//   v_wmma_i32_16x16x16_iu8 (0x44)  v_wmma_f32_16x16x16_f16 (0x40, control)                        [gfx11 + gfx12]
//   v_swmmac_* (0x50..0x5a)                                                                         [gfx12 only]
// The program records raw operand and result bits only. It contains NO arithmetic model: every conclusion is
// drawn offline by analyze.py. Mapping controls (identity A / identity B / C passthrough) are part of every
// WMMA input set so a wrong fragment-layout assumption is detected, not silently absorbed.
//
// Build (CPU only, no GPU launch):  build.sh <gfx1151|gfx1201> [outdir]
// Run (inside a bounded fabric hold):  numerics_probe <outdir> <suite[,suite...]|all>
//   suites: identity dot wmma iu8 swmmac_map swmmac_num swmmac_valid; opt-in (not in `all`): div_scale scalar_udiv maxmin
// Output per test: <name>.json (schema/meta, mode request + MODE readback) and <name>.out.bin / <name>.in.bin.
#include <hip/hip_runtime.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>

#if defined(NP_GFX1201)
#define NP_GFX12 1
#elif defined(NP_GFX1151)
#define NP_GFX12 0
#else
#error "define NP_GFX1201 or NP_GFX1151"
#endif
#if defined(__HIP_DEVICE_COMPILE__)
#if NP_GFX12 && !defined(__gfx1201__)
#error "NP_GFX1201 requires --offload-arch=gfx1201"
#elif !NP_GFX12 && !defined(__gfx1151__)
#error "NP_GFX1151 requires --offload-arch=gfx1151"
#endif
#endif
#define HIP(C) do{hipError_t e_=(C);if(e_!=hipSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#C,hipGetErrorString(e_));exit(2);}}while(0)

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
#if NP_GFX12
constexpr const char* ARCH = "gfx1201";
#else
constexpr const char* ARCH = "gfx1151";
#endif

// ---------------------------------------------------------------------------------------------- MODE control
constexpr uint32_t MODE_KEEP = 0xFFFFFFFFu;  // do not touch MODE: measures the launch default
__device__ __forceinline__ void set_mode(uint32_t m) {
  if (m == MODE_KEEP) return;
  uint32_t s = __builtin_amdgcn_readfirstlane(m);
  asm volatile("s_setreg_b32 hwreg(HW_REG_MODE, 0, 8), %0\n\ts_nop 7" ::"s"(s) : "memory");
}
__device__ __forceinline__ uint32_t get_mode() {
  uint32_t r;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 32)" : "=s"(r));
  return r;
}
// fp_round[1:0]=f32, [3:2]=f16/f64 ; fp_denorm[5:4]=f32, [7:6]=f16/f64 (bit0 of a pair = input, bit1 = output)
static const uint32_t MODES[] = {MODE_KEEP, 0x00, 0xC0, 0x30, 0xF1, 0xF2, 0xF3, 0xFC};
constexpr int NMODES = sizeof(MODES) / sizeof(MODES[0]);

// ---------------------------------------------------------------------------------------------- host utils
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  uint32_t u32() { return uint32_t(next() >> 32); }
  uint32_t below(uint32_t n) { return uint32_t(((next() >> 32) * uint64_t(n)) >> 32); }
};
static std::string OUT;
static std::vector<std::string> INDEX;
static void write_file(const std::string& path, const void* p, size_t n) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
  if (n && fwrite(p, 1, n, f) != n) { fprintf(stderr, "short write %s\n", path.c_str()); exit(2); }
  fclose(f);
}
static void write_text(const std::string& path, const std::string& s) { write_file(path, s.data(), s.size()); }
static std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char* f, ...) {
  char b[4096];
  va_list ap;
  va_start(ap, f);
  vsnprintf(b, sizeof b, f, ap);
  va_end(ap);
  return b;
}
template <class T> static std::string hexlist(const std::vector<T>& v, const char* f) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + fmt(f, (unsigned)v[i]);
  return s + "]";
}
template <class T> struct DevBuf {
  T* p = nullptr;
  size_t n = 0;
  explicit DevBuf(size_t count) : n(count) { HIP(hipMalloc(&p, count * sizeof(T) + 16)); HIP(hipMemset(p, 0, count * sizeof(T) + 16)); }
  explicit DevBuf(const std::vector<T>& v) : DevBuf(v.size()) { HIP(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); }
  ~DevBuf() { (void)hipFree(p); }
  std::vector<T> get() const { std::vector<T> v(n); HIP(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return v; }
};
static uint32_t popmode(DevBuf<uint32_t>& mo) { return mo.get()[0]; }

// ---------------------------------------------------------------------------------------------- value sets
// bf16 / f16 / f32 special values. Index positions are part of the contract (analysis reads the lists from the .json).
static const std::vector<uint16_t> BF16_S = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x0040, 0x007f, 0x807f, 0x0080, 0x8080, 0x0081, 0x3380, 0x3f80, 0xbf80, 0x3f81, 0x3f7f,
    0x4000, 0x4040, 0x4100, 0x4b80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7f81, 0x7fff, 0x3f00, 0x0100, 0x4b00, 0xc040};
static const std::vector<uint16_t> F16_S = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x0200, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x0401, 0x3c00, 0xbc00, 0x3c01, 0x3bff, 0x4000,
    0x4200, 0x4800, 0x5640, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x7c01, 0x7dff, 0x7e01, 0x1400, 0x0100, 0x6400, 0x7800};
static const std::vector<uint32_t> F32_S = {
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x00000100, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x00800001, 0x33800000,
    0x34000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x3f7fffff, 0x40000000, 0x40400000, 0x4b000000, 0x4b800000, 0x4b800001, 0xcb800000,
    0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001, 0x7fc0abcd, 0x3effffff, 0x5f000000};
static const std::vector<uint16_t> BF16_R = {0x0000, 0x8000, 0x0001, 0x807f, 0x0080, 0x3f80, 0xbf80, 0x3f81, 0x4040, 0x4100, 0x3380, 0x4b80, 0x7f7f, 0x7f80, 0xff80, 0x7fc0};
static const std::vector<uint16_t> F16_R = {0x0000, 0x8000, 0x0001, 0x83ff, 0x0400, 0x3c00, 0xbc00, 0x3c01, 0x4200, 0x4800, 0x1400, 0x6400, 0x7bff, 0x7c00, 0xfc00, 0x7e00};
static const std::vector<uint32_t> F32_R6 = {0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x00000001, 0x4b800000};
static const std::vector<uint32_t> I32_S = {0, 1, 0xffffffffu, 0x7fffffffu, 0x80000000u, 0x7ffffffeu, 0x80000001u, 0x7ffe7960u, 0x80018000u,
                                            0x7fff0000u, 0x80010000u, 0x40000000u, 0xc0000000u, 0x00010000u, 0xffff0000u, 12345};

// random value of a 16-bit/32-bit float format around a base exponent
static uint32_t rnd_float(Rng& g, int ebits, int fbits, int base_exp, int spread) {
  int emax = (1 << ebits) - 1, bias = emax >> 1;
  uint32_t sign = g.u32() & 1;
  int e = base_exp + int(g.below(2 * spread + 1)) - spread;
  if (e < 0) e = 0;
  if (e > emax - 1) e = emax - 1;
  uint32_t mask = (1u << fbits) - 1, f = g.u32() & mask;
  uint32_t r = g.below(8);
  if (r == 0) f = 0;
  else if (r == 1) f = 1u << g.below(fbits);
  else if (r == 2) f = mask;
  else if (r == 3) f &= ~((1u << (fbits / 2)) - 1);
  (void)bias;
  return (sign << (ebits + fbits)) | (uint32_t(e) << fbits) | f;
}
static uint32_t rnd_c(Rng& g, int base_exp, int spread) {
  if (g.below(10) == 0) return F32_S[g.below(F32_S.size())];
  return rnd_float(g, 8, 23, base_exp, spread);
}

// ============================================================================================== DOT
// ops: 0 dot2_f32_f16 | 1 dot2_f32_bf16 | 2 dot4_i32_iu8 | 3 dot8_i32_iu4
#define DOTF_ASM(MN, M) asm volatile(MN " %0, %1, %2, %3" M : "=v"(d) : "v"(a), "v"(b), "v"(c))
#define DOTF_BODY(MN)                                                                                                    \
  if constexpr (V == 0) { DOTF_ASM(MN, ""); }                                                                            \
  else if constexpr (V == 1) { DOTF_ASM(MN, " clamp"); }                                                                 \
  else if constexpr (V == 2) { DOTF_ASM(MN, " neg_lo:[1,0,0]"); }                                                        \
  else if constexpr (V == 3) { DOTF_ASM(MN, " neg_lo:[0,1,0]"); }                                                        \
  else if constexpr (V == 4) { DOTF_ASM(MN, " neg_lo:[0,0,1]"); }                                                        \
  else if constexpr (V == 5) { DOTF_ASM(MN, " neg_hi:[1,0,0]"); }                                                        \
  else if constexpr (V == 6) { DOTF_ASM(MN, " neg_hi:[0,1,0]"); }                                                        \
  else if constexpr (V == 7) { DOTF_ASM(MN, " neg_hi:[0,0,1]"); }                                                        \
  else if constexpr (V == 8) { DOTF_ASM(MN, " op_sel:[1,0,0]"); }                                                        \
  else if constexpr (V == 9) { DOTF_ASM(MN, " op_sel:[0,1,0]"); }                                                        \
  else if constexpr (V == 10) { DOTF_ASM(MN, " op_sel:[1,1,0]"); }                                                       \
  else if constexpr (V == 11) { DOTF_ASM(MN, " op_sel_hi:[0,1,1]"); }                                                    \
  else if constexpr (V == 12) { DOTF_ASM(MN, " op_sel_hi:[1,0,1]"); }                                                    \
  else if constexpr (V == 13) { DOTF_ASM(MN, " op_sel_hi:[0,0,1]"); }                                                    \
  else if constexpr (V == 14) { DOTF_ASM(MN, " neg_lo:[1,1,1] neg_hi:[1,1,1]"); }
static const char* DOTF_MODS[] = {"", "clamp", "neg_lo:[1,0,0]", "neg_lo:[0,1,0]", "neg_lo:[0,0,1]", "neg_hi:[1,0,0]", "neg_hi:[0,1,0]",
                                  "neg_hi:[0,0,1]", "op_sel:[1,0,0]", "op_sel:[0,1,0]", "op_sel:[1,1,0]", "op_sel_hi:[0,1,1]",
                                  "op_sel_hi:[1,0,1]", "op_sel_hi:[0,0,1]", "neg_lo:[1,1,1] neg_hi:[1,1,1]"};
constexpr int NDOTF = 15;
#define DOTI_BODY(MN)                                                                                                    \
  if constexpr (V == 0) { DOTF_ASM(MN, ""); }                                                                            \
  else if constexpr (V == 1) { DOTF_ASM(MN, " clamp"); }                                                                 \
  else if constexpr (V == 2) { DOTF_ASM(MN, " neg_lo:[1,0,0]"); }                                                        \
  else if constexpr (V == 3) { DOTF_ASM(MN, " neg_lo:[0,1,0]"); }                                                        \
  else if constexpr (V == 4) { DOTF_ASM(MN, " neg_lo:[1,1,0]"); }                                                        \
  else if constexpr (V == 5) { DOTF_ASM(MN, " neg_lo:[1,0,0] clamp"); }                                                  \
  else if constexpr (V == 6) { DOTF_ASM(MN, " neg_lo:[0,1,0] clamp"); }                                                  \
  else if constexpr (V == 7) { DOTF_ASM(MN, " neg_lo:[1,1,0] clamp"); }                                                  \
  else if constexpr (V == 8) { DOTF_ASM(MN, " neg_lo:[0,0,1]"); }                                                        \
  else if constexpr (V == 9) { DOTF_ASM(MN, " neg_hi:[1,0,0]"); }
static const char* DOTI_MODS[] = {"", "clamp", "neg_lo:[1,0,0]", "neg_lo:[0,1,0]", "neg_lo:[1,1,0]", "neg_lo:[1,0,0] clamp",
                                  "neg_lo:[0,1,0] clamp", "neg_lo:[1,1,0] clamp", "neg_lo:[0,0,1]", "neg_hi:[1,0,0]"};
constexpr int NDOTI = 10;
static const char* DOT_NAME[] = {"v_dot2_f32_f16", "v_dot2_f32_bf16", "v_dot4_i32_iu8", "v_dot8_i32_iu4"};

template <int OP, int V>
__global__ void dot_k(const uint32_t* in, uint32_t* out, uint32_t n, uint32_t mode, uint32_t* modeout) {
  set_mode(mode);
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) modeout[0] = get_mode();
  if (i >= n) return;
  uint32_t a = in[3 * size_t(i)], b = in[3 * size_t(i) + 1], c = in[3 * size_t(i) + 2], d;
  if constexpr (OP == 0) { DOTF_BODY("v_dot2_f32_f16") }
  else if constexpr (OP == 1) { DOTF_BODY("v_dot2_f32_bf16") }
  else if constexpr (OP == 2) { DOTI_BODY("v_dot4_i32_iu8") }
  else { DOTI_BODY("v_dot8_i32_iu4") }
  out[i] = d;
}
template <int OP, int N> static void dot_launch(int v, DevBuf<uint32_t>& in, DevBuf<uint32_t>& out, uint32_t n, uint32_t mode, DevBuf<uint32_t>& mo) {
  if constexpr (N > 0) {
    if (v == N - 1) { dot_k<OP, N - 1><<<(n + 255) / 256, 256>>>(in.p, out.p, n, mode, mo.p); return; }
    dot_launch<OP, N - 1>(v, in, out, n, mode, mo);
  }
}

static void pack_pair(std::vector<uint32_t>& v, uint32_t x, uint32_t y, uint32_t z) { v.push_back(x); v.push_back(y); v.push_back(z); }

// case sets: returns flat (a,b,c) triples
static std::vector<uint32_t> dot_set(int op, bool red, uint64_t seed) {
  std::vector<uint32_t> v;
  Rng g(seed);
  if (op <= 1) {
    const std::vector<uint16_t>& S = op == 0 ? F16_S : BF16_S;
    const int ebits = op == 0 ? 5 : 8, fbits = op == 0 ? 10 : 7;
    // exhaustive Cartesian over 14 specials in (a0,b0,a1,b1) and 24 f32 specials in C
    static const int pick[14] = {0, 1, 2, 3, 8, 11, 12, 13, 14, 16, 19, 22, 23, 24};
    std::vector<uint16_t> s14;
    for (int i : pick) s14.push_back(S[i]);
    // spec index set used for C: first 24 of F32_S
    uint32_t stride = red ? 31 : 1, k = 0;
    for (uint16_t a0 : s14) for (uint16_t b0 : s14) for (uint16_t a1 : s14) for (uint16_t b1 : s14)
      for (int ci = 0; ci < 24; ci++) {
        if (k++ % stride) continue;
        pack_pair(v, a0 | (uint32_t(a1) << 16), b0 | (uint32_t(b1) << 16), F32_S[ci]);
      }
    uint32_t nrand = red ? 70000 : 400000;
    for (uint32_t i = 0; i < nrand; i++) {
      int base = (op == 0 ? 15 : 127) + int(g.below(25)) - 12, spread = int(g.below(4)) * 3;
      auto el = [&]() { return g.below(8) == 0 ? uint32_t(S[g.below(S.size())]) : rnd_float(g, ebits, fbits, base, spread); };
      uint32_t a = el() | (el() << 16), b = el() | (el() << 16);
      pack_pair(v, a, b, rnd_c(g, 127 + (base - (op == 0 ? 15 : 127)) * 2 + int(g.below(9)) - 4, 3));
    }
  } else {
    const bool i4 = op == 3;
    std::vector<uint32_t> P = {0x00000000, 0x01010101, 0x7f7f7f7f, 0x80808080, 0xffffffff, 0x81818181, 0x7f808081, 0x80ff017f, 0xff00ff00,
                               0x7f7f7f80, 0x00ff00ff, 0x7fffff80};
    if (i4) P = {0x00000000, 0x11111111, 0x77777777, 0x88888888, 0xffffffff, 0x99999999, 0x7f8081ff, 0x8f7f0fff, 0xf0f0f0f0, 0x77777778, 0x0f0f0f0f, 0x7f7f7f80};
    for (uint32_t a : P) for (uint32_t b : P) for (uint32_t c : I32_S) pack_pair(v, a, b, c);
    uint32_t nrand = red ? 70000 : 400000;
    for (uint32_t i = 0; i < nrand; i++) {
      auto el = [&]() { uint32_t r = g.below(6); return r == 0 ? P[g.below(P.size())] : r == 1 ? (g.u32() & 0x80808080u) | (g.u32() & 0x03030303u) : g.u32(); };
      uint32_t c = g.below(4) == 0 ? I32_S[g.below(I32_S.size())] : g.below(2) ? g.u32() : uint32_t(int32_t(g.u32()) >> g.below(32));
      pack_pair(v, el(), el(), c);
    }
  }
  return v;
}

static void record(const std::string& name, const std::string& json) {
  write_text(OUT + "/" + name + ".json", json);
  INDEX.push_back(name);
}

static void suite_dot() {
  for (int op = 0; op < 4; op++) {
    int nv = op < 2 ? NDOTF : NDOTI;
    for (int red = 0; red < 2; red++) {
      std::vector<uint32_t> in = dot_set(op, red, 0xd07 + op * 17 + red);
      uint32_t n = in.size() / 3;
      std::string setname = red ? "red" : "full";
      std::string inname = fmt("dot.%s.%s.in", DOT_NAME[op], setname.c_str());
      write_file(OUT + "/" + inname + ".bin", in.data(), in.size() * 4);
      record(inname, fmt("{\"kind\":\"dot_inputs\",\"arch\":\"%s\",\"op\":\"%s\",\"set\":\"%s\",\"cases\":%u,\"record\":\"u32 a,b,c\"}", ARCH, DOT_NAME[op], setname.c_str(), n));
      DevBuf<uint32_t> din(in), dout(n), mo(1);
      // (full: V0 over all modes) (red: every modifier variant at default mode, V0 over all modes)
      for (int v = 0; v < nv; v++)
        for (int m = 0; m < NMODES; m++) {
          if (v && (!red || m)) continue;
          switch (op) {
            case 0: dot_launch<0, NDOTF>(v, din, dout, n, MODES[m], mo); break;
            case 1: dot_launch<1, NDOTF>(v, din, dout, n, MODES[m], mo); break;
            case 2: dot_launch<2, NDOTI>(v, din, dout, n, MODES[m], mo); break;
            default: dot_launch<3, NDOTI>(v, din, dout, n, MODES[m], mo); break;
          }
          HIP(hipGetLastError());
          HIP(hipDeviceSynchronize());
          auto res = dout.get();
          std::string name = fmt("dot.%s.%s.v%d.m%d.out", DOT_NAME[op], setname.c_str(), v, m);
          write_file(OUT + "/" + name + ".bin", res.data(), n * 4);
          record(name, fmt("{\"kind\":\"dot_result\",\"arch\":\"%s\",\"op\":\"%s\",\"set\":\"%s\",\"cases\":%u,\"variant\":%d,\"mods\":\"%s\",\"mode_index\":%d,\"mode_request\":\"0x%08x\",\"mode_readback\":\"0x%08x\",\"inputs\":\"%s\"}",
                           ARCH, DOT_NAME[op], setname.c_str(), n, v, op < 2 ? DOTF_MODS[v] : DOTI_MODS[v], m, MODES[m], popmode(mo), inname.c_str()));
        }
    }
  }
}

// ============================================================================================== WMMA (dense)
// One wave32 per case; logical matrices in, raw D registers out.
struct WCase { uint16_t A[256]; uint16_t Bt[256]; uint32_t C[256]; };  // A[r*16+k], Bt[c*16+k], C[r*16+c]
// kinds: 0 bf16->f32 (0x41) | 1 bf16->bf16 (0x43) | 2 f16->f16 (0x42) | 3 iu8->i32 (0x44) | 4 f16->f32 (0x40)
static const char* WKIND[] = {"v_wmma_f32_16x16x16_bf16", "v_wmma_bf16_16x16x16_bf16", "v_wmma_f16_16x16x16_f16", "v_wmma_i32_16x16x16_iu8", "v_wmma_f32_16x16x16_f16"};
constexpr bool wk_out16(int k) { return k == 1 || k == 2; }
constexpr int NWV_F = NP_GFX12 ? 8 : 9;  // float variants (gfx11 has the extra op_sel[2] variant for 16-bit output)
constexpr int NWV_I = 8;

#if NP_GFX12
typedef v4i FragF; typedef v2i FragI;
constexpr int NVO16 = 4;
#else
typedef v8i FragF; typedef v4i FragI;
constexpr int NVO16 = 8;
#endif
#define WA(MN, M) asm volatile("s_nop 4\n\t" MN " %[d], %[a], %[b], %[d]" M "\n\ts_nop 7" : [d] "+v"(d) : [a] "v"(a), [b] "v"(b) : "memory")
#define WFBODY(MN)                                                                                          \
  if constexpr (V == 0) { WA(MN, ""); }                                                                     \
  else if constexpr (V == 1) { WA(MN, " neg_lo:[1,0,0]"); }                                                 \
  else if constexpr (V == 2) { WA(MN, " neg_lo:[0,1,0]"); }                                                 \
  else if constexpr (V == 3) { WA(MN, " neg_lo:[0,0,1]"); }                                                 \
  else if constexpr (V == 4) { WA(MN, " neg_hi:[1,0,0]"); }                                                 \
  else if constexpr (V == 5) { WA(MN, " neg_hi:[0,1,0]"); }                                                 \
  else if constexpr (V == 6) { WA(MN, " neg_hi:[0,0,1]"); }                                                 \
  else if constexpr (V == 7) { WA(MN, " neg_lo:[1,1,1] neg_hi:[1,1,1]"); }
#define WIBODY(MN)                                                                                          \
  if constexpr (V == 0) { WA(MN, ""); }                                                                     \
  else if constexpr (V == 1) { WA(MN, " neg_lo:[1,0,0]"); }                                                 \
  else if constexpr (V == 2) { WA(MN, " neg_lo:[0,1,0]"); }                                                 \
  else if constexpr (V == 3) { WA(MN, " neg_lo:[1,1,0]"); }                                                 \
  else if constexpr (V == 4) { WA(MN, " clamp"); }                                                          \
  else if constexpr (V == 5) { WA(MN, " neg_lo:[1,0,0] clamp"); }                                           \
  else if constexpr (V == 6) { WA(MN, " neg_lo:[0,1,0] clamp"); }                                           \
  else if constexpr (V == 7) { WA(MN, " neg_lo:[1,1,0] clamp"); }
static const char* WFMODS[] = {"", "neg_lo:[1,0,0]", "neg_lo:[0,1,0]", "neg_lo:[0,0,1]", "neg_hi:[1,0,0]", "neg_hi:[0,1,0]", "neg_hi:[0,0,1]", "neg_lo:[1,1,1] neg_hi:[1,1,1]", "op_sel:[0,0,1]"};
static const char* WIMODS[] = {"", "neg_lo:[1,0,0]", "neg_lo:[0,1,0]", "neg_lo:[1,1,0]", "clamp", "neg_lo:[1,0,0] clamp", "neg_lo:[0,1,0] clamp", "neg_lo:[1,1,0] clamp"};

template <int KIND, int V>
__global__ void __launch_bounds__(32) wmma_k(const WCase* cs, uint32_t* out, uint32_t n, uint32_t mode, uint32_t* modeout) {
  set_mode(mode);
  const uint32_t w = blockIdx.x, l = threadIdx.x, r = l & 15, h = l >> 4;
  if (w == 0 && l == 0) modeout[0] = get_mode();
  if (w >= n) return;
  const WCase& cc = cs[w];
  if constexpr (KIND == 3) {
    FragI a, b;
    constexpr int NW = sizeof(FragI) / 4;
    for (int i = 0; i < NW; i++) {
      int k0 = NP_GFX12 ? int(8 * h) + 4 * i : 4 * i;
      a[i] = int(uint32_t(cc.A[r * 16 + k0] & 255) | (uint32_t(cc.A[r * 16 + k0 + 1] & 255) << 8) | (uint32_t(cc.A[r * 16 + k0 + 2] & 255) << 16) | (uint32_t(cc.A[r * 16 + k0 + 3] & 255) << 24));
      b[i] = int(uint32_t(cc.Bt[r * 16 + k0] & 255) | (uint32_t(cc.Bt[r * 16 + k0 + 1] & 255) << 8) | (uint32_t(cc.Bt[r * 16 + k0 + 2] & 255) << 16) | (uint32_t(cc.Bt[r * 16 + k0 + 3] & 255) << 24));
    }
    v8i d;
    for (int j = 0; j < 8; j++) d[j] = int(cc.C[(NP_GFX12 ? 8 * h + j : 2 * j + h) * 16 + r]);
    WIBODY("v_wmma_i32_16x16x16_iu8")
    for (int j = 0; j < 8; j++) out[(size_t(w) * 8 + j) * 32 + l] = uint32_t(d[j]);
  } else {
    FragF a, b;
    constexpr int NW = sizeof(FragF) / 4;
    for (int i = 0; i < NW; i++) {
      int k0 = NP_GFX12 ? int(8 * h) + 2 * i : 2 * i;
      a[i] = int(uint32_t(cc.A[r * 16 + k0]) | (uint32_t(cc.A[r * 16 + k0 + 1]) << 16));
      b[i] = int(uint32_t(cc.Bt[r * 16 + k0]) | (uint32_t(cc.Bt[r * 16 + k0 + 1]) << 16));
    }
    if constexpr (!wk_out16(KIND)) {
      v8i d;
      for (int j = 0; j < 8; j++) d[j] = int(cc.C[(NP_GFX12 ? 8 * h + j : 2 * j + h) * 16 + r]);
      if constexpr (KIND == 0) { WFBODY("v_wmma_f32_16x16x16_bf16") } else { WFBODY("v_wmma_f32_16x16x16_f16") }
      for (int j = 0; j < 8; j++) out[(size_t(w) * 8 + j) * 32 + l] = uint32_t(d[j]);
    } else {
#if NP_GFX12
      v4i d;
      for (int j = 0; j < 4; j++) d[j] = int((cc.C[(8 * h + 2 * j) * 16 + r] & 0xffff) | (cc.C[(8 * h + 2 * j + 1) * 16 + r] << 16));
      if constexpr (KIND == 1) { WFBODY("v_wmma_bf16_16x16x16_bf16") } else { WFBODY("v_wmma_f16_16x16x16_f16") }
      for (int j = 0; j < 4; j++) out[(size_t(w) * NVO16 + j) * 32 + l] = uint32_t(d[j]);
#else
      v8i d;
      for (int j = 0; j < 8; j++) {
        uint32_t v = cc.C[(2 * j + h) * 16 + r] & 0xffff;
        d[j] = int(V == 8 ? ((v << 16) | 0x5a5au) : (v | 0xa5a50000u));  // opposite half poisoned: reveals preserve/ignore/zero
      }
      if constexpr (V == 8) {
        if constexpr (KIND == 1) { WA("v_wmma_bf16_16x16x16_bf16", " op_sel:[0,0,1]"); } else { WA("v_wmma_f16_16x16x16_f16", " op_sel:[0,0,1]"); }
      } else {
        if constexpr (KIND == 1) { WFBODY("v_wmma_bf16_16x16x16_bf16") } else { WFBODY("v_wmma_f16_16x16x16_f16") }
      }
      for (int j = 0; j < 8; j++) out[(size_t(w) * NVO16 + j) * 32 + l] = uint32_t(d[j]);
#endif
    }
  }
}
template <int KIND, int N> static void wmma_launch(int v, DevBuf<WCase>& in, DevBuf<uint32_t>& out, uint32_t n, uint32_t mode, DevBuf<uint32_t>& mo) {
  if constexpr (N > 0) {
    if (v == N - 1) { wmma_k<KIND, N - 1><<<n, 32>>>(in.p, out.p, n, mode, mo.p); return; }
    wmma_launch<KIND, N - 1>(v, in, out, n, mode, mo);
  }
}
static void wmma_dispatch(int kind, int v, DevBuf<WCase>& in, DevBuf<uint32_t>& out, uint32_t n, uint32_t mode, DevBuf<uint32_t>& mo) {
  switch (kind) {
    case 0: wmma_launch<0, 8>(v, in, out, n, mode, mo); break;
    case 1: wmma_launch<1, NWV_F>(v, in, out, n, mode, mo); break;
    case 2: wmma_launch<2, NWV_F>(v, in, out, n, mode, mo); break;
    case 3: wmma_launch<3, NWV_I>(v, in, out, n, mode, mo); break;
    default: wmma_launch<4, 8>(v, in, out, n, mode, mo); break;
  }
}

// ----------------------------------------------------------------------------------- dense case generators
struct WSet { std::vector<WCase> cs; uint32_t n_control = 3, n_sub = 0; std::string layout; };
static WCase zero_case() { WCase c; memset(&c, 0, sizeof c); return c; }


static WSet make_float_set(int kind, uint64_t seed) {
  const bool bf = kind <= 1;
  const std::vector<uint16_t>& S = bf ? BF16_S : F16_S;
  const std::vector<uint16_t>& R = bf ? BF16_R : F16_R;
  const int ebits = bf ? 8 : 5, fbits = bf ? 7 : 10, bias = bf ? 127 : 15;
  const bool o16 = wk_out16(kind);
  WSet s;
  // controls: D=B (A=I), D=C (A=B=0), D=A (B=I). Unique small integers: exact in bf16 (<=255) and f16.
  auto num16 = [&](int v) -> uint16_t {  // positive integer 1..255 in this format
    uint32_t e = 31 - __builtin_clz(v), f = (v - (1u << e)) << (fbits - e);
    return uint16_t(((e + bias) << fbits) | f);
  };
  auto cnum = [&](int v) -> uint32_t {  // same integer as accumulator format
    if (o16) return num16(v);
    float f = float(v); uint32_t u; memcpy(&u, &f, 4); return u;
  };
  { WCase c = zero_case(); for (int i = 0; i < 16; i++) c.A[i * 16 + i] = num16(1); for (int k = 0; k < 16; k++) for (int j = 0; j < 16; j++) c.Bt[j * 16 + k] = num16(1 + ((k * 16 + j) % 250)); s.cs.push_back(c); }
  { WCase c = zero_case(); for (int r = 0; r < 16; r++) for (int j = 0; j < 16; j++) c.C[r * 16 + j] = cnum(1 + ((r * 16 + j) % 250)); s.cs.push_back(c); }
  { WCase c = zero_case(); for (int j = 0; j < 16; j++) c.Bt[j * 16 + j] = num16(1); for (int r = 0; r < 16; r++) for (int k = 0; k < 16; k++) c.A[r * 16 + k] = num16(1 + ((r * 16 + k) % 250)); s.cs.push_back(c); }
  // ---- F1: single product at every physical K position; Cartesian over 32 x 32 specials x 32 C specials
  std::vector<uint32_t> Cs = F32_S;
  if (o16) { Cs.assign(S.begin(), S.end()); }
  for (int kp = 0; kp < 16; kp++)
    for (size_t ab = 0; ab < S.size(); ab += 16)
      for (size_t bb = 0; bb < S.size(); bb += 16)
        for (uint32_t cv : Cs) {
          WCase c = zero_case();
          for (int i = 0; i < 16; i++) { c.A[i * 16 + kp] = S[ab + i]; c.Bt[i * 16 + kp] = S[bb + i]; }
          for (int i = 0; i < 256; i++) c.C[i] = cv;
          s.cs.push_back(c);
        }
  // ---- F4: targeted subnormal / tiny accumulations (all-K subnormal rows x sparse small-normal columns)
  {
    std::vector<uint16_t> sub, nz;
    for (uint16_t v : (bf ? std::vector<uint16_t>{0x0001, 0x8001, 0x0002, 0x0003, 0x0040, 0x007f, 0x807f, 0x0055, 0x0010, 0x8002, 0x0004, 0x0008, 0x0020, 0x0007, 0x0070, 0x8040}
                          : std::vector<uint16_t>{0x0001, 0x8001, 0x0002, 0x0003, 0x0200, 0x03ff, 0x83ff, 0x0155, 0x0010, 0x8002, 0x0004, 0x0008, 0x0100, 0x0007, 0x0300, 0x8200})) sub.push_back(v);
    std::vector<uint16_t> big = bf ? std::vector<uint16_t>{0x4040, 0x4100, 0x3f80, 0x4000, 0xc040, 0x3f81, 0x4b80, 0x4b00, 0x3380, 0x0080, 0x0100, 0x7f7f, 0x3f00, 0x4080, 0xc100, 0x3fc0}
                                   : std::vector<uint16_t>{0x4200, 0x4800, 0x3c00, 0x4000, 0xc200, 0x3c01, 0x6400, 0x5c00, 0x1400, 0x0400, 0x0800, 0x7bff, 0x3800, 0x4400, 0xc800, 0x3e00};
    static const int posets[6][3] = {{0, 1, -1}, {0, 4, -1}, {0, 8, -1}, {3, 4, -1}, {15, -1, -1}, {0, 1, 2}};
    for (uint32_t cv : Cs)
      for (int ps = 0; ps < 6; ps++)
        for (int vb = 0; vb < 4; vb++) {
          WCase c = zero_case();
          for (int r = 0; r < 16; r++) for (int k = 0; k < 16; k++) c.A[r * 16 + k] = sub[(r + (vb & 1) * 5) % 16];
          for (int j = 0; j < 16; j++) for (int t = 0; t < 3; t++) if (posets[ps][t] >= 0) c.Bt[j * 16 + posets[ps][t]] = big[(j + t * 3 + vb * 7) % 16];
          for (int i = 0; i < 256; i++) c.C[i] = cv;
          if (vb >= 2) for (int j = 0; j < 16; j++) for (int k = 0; k < 16; k++) if (!c.Bt[j * 16 + k] && (vb == 3)) c.Bt[j * 16 + k] = sub[(j + k) % 16];
          s.cs.push_back(c);
        }
  }
  // ---- F5: accumulator-tail window sweep. Rows are all-K subnormal values, column c carries ONE normal power-of-two
  //          (mantissa 1.0 or 1.5) at K position k0 with its exponent sweeping every normal exponent; C = tiny +-subnormal.
  //          Discriminates the grid-vs-C-exponent distance at which a negative accumulator tail is floored
  //          (e.g. A=[0x0001]x16, C=0x80000001, B[0] in {4100, 4B80, 4C00, 7F00}).
  {
    const std::vector<uint16_t> subs = bf ? std::vector<uint16_t>{0x0001, 0x8001, 0x0003, 0x0040, 0x007f, 0x8002, 0x0010, 0x0002, 0x8040, 0x0004, 0x0005, 0x0008, 0x0020, 0x8003, 0x0006, 0x0007}
                                          : std::vector<uint16_t>{0x0001, 0x8001, 0x0003, 0x0200, 0x03ff, 0x8002, 0x0010, 0x0002, 0x8200, 0x0004, 0x0005, 0x0008, 0x0100, 0x8003, 0x0006, 0x0007};
    const std::vector<uint32_t> tinyC = {0x80000001, 0x80000002, 0x807fffff, 0x80000100, 0x00000001, 0x80800000, 0x00800001, 0x80000003};
    const std::vector<uint16_t> tinyC16 = bf ? std::vector<uint16_t>{0x8001, 0x8002, 0x807f, 0x8040, 0x0001, 0x8080, 0x0081, 0x8003}
                                             : std::vector<uint16_t>{0x8001, 0x8002, 0x83ff, 0x8200, 0x0001, 0x8400, 0x0401, 0x8003};
    const int emax = (1 << ebits) - 1;
    for (int k0 : {0, 5, 12})
      for (int mant = 0; mant < 2; mant++)
        for (int e0 = 1; e0 < emax; e0 += 16)
          for (size_t ci = 0; ci < tinyC.size(); ci++) {
            WCase c = zero_case();
            for (int r = 0; r < 16; r++) for (int k = 0; k < 16; k++) c.A[r * 16 + k] = subs[r];
            for (int j = 0; j < 16; j++) {
              int e = std::min(e0 + j, emax - 1);
              c.Bt[j * 16 + k0] = uint16_t((e << fbits) | (mant ? 1u << (fbits - 1) : 0));
            }
            for (int i = 0; i < 256; i++) c.C[i] = o16 ? uint32_t(tinyC16[ci]) : tinyC[ci];
            s.cs.push_back(c);
          }
  }
  // ---- F3: random full rows (2000 waves in the `sub` prefix)
  Rng g(seed);
  auto f3 = [&](uint32_t count) {
    for (uint32_t w = 0; w < count; w++) {
      WCase c = zero_case();
      int base = bias + int(g.below(2 * bias / 2)) - bias / 2, spread = int(g.below(5)) * 3;
      auto el = [&]() -> uint16_t {
        uint32_t r = g.below(10);
        if (r < 1) return (g.u32() & 1) << 15;
        if (r < 2) return S[g.below(S.size())];
        return uint16_t(rnd_float(g, ebits, fbits, base, spread));
      };
      // exclude NaN/Inf for ~85% of waves so accumulation is exercised, keep all classes elsewhere
      bool finite_only = g.below(100) < 85;
      auto fix = [&](uint16_t v) -> uint16_t { if (finite_only && ((v >> fbits) & ((1 << ebits) - 1)) == (1u << ebits) - 1) return v & ~(1u << (fbits)); return v; };
      for (int i = 0; i < 256; i++) { c.A[i] = fix(el()); c.Bt[i] = fix(el()); }
      for (int i = 0; i < 256; i++) {
        if (o16) c.C[i] = fix(g.below(10) == 0 ? S[g.below(S.size())] : uint16_t(rnd_float(g, ebits, fbits, base + int(g.below(2 * spread + 1)) - spread + (bf ? 0 : 0), 2)));
        else c.C[i] = rnd_c(g, 127 + (base - bias) * 2 + int(g.below(15)) - 7, 4);
        if (!o16 && finite_only && ((c.C[i] >> 23) & 255) == 255) c.C[i] &= ~(1u << 30);
      }
      s.cs.push_back(c);
    }
  };
  f3(2000);
  s.n_sub = s.cs.size();
  // ---- F2: two products at selected K position pairs, full Cartesian over 16 values for (a0,a1,b0,b1)
  static const int kp[10][2] = {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {3, 4}, {7, 8}, {0, 15}, {12, 13}, {8, 9}, {4, 12}};
  for (int p = 0; p < 10; p++)
    for (int ci = 0; ci < 6; ci++)
      for (int a1 = 0; a1 < 16; a1++)
        for (int b1 = 0; b1 < 16; b1++) {
          WCase c = zero_case();
          for (int i = 0; i < 16; i++) { c.A[i * 16 + kp[p][0]] = R[i]; c.A[i * 16 + kp[p][1]] = R[a1]; c.Bt[i * 16 + kp[p][0]] = R[i]; c.Bt[i * 16 + kp[p][1]] = R[b1]; }
          uint32_t cv = o16 ? uint32_t(R[(ci * 3) % 16]) : F32_R6[ci];
          for (int i = 0; i < 256; i++) c.C[i] = cv;
          s.cs.push_back(c);
        }
  f3(8000);
  // ---- F6: mid-step precision. Two products in DIFFERENT dot steps (K positions kA, kB): p0 = V[r]*V[c] (dense 16-bit
  //          significands, every row/column pair), p1 = +-2^-s with s sweeping; C = +0. Where a carry/borrow of p1 into p0
  //          appears reveals how many significand bits the inter-step accumulator keeps and how it is rounded.
  {
    static const int kc[12][2] = {{0, 4}, {0, 8}, {0, 12}, {4, 8}, {4, 12}, {8, 12}, {1, 5}, {2, 13}, {3, 4}, {3, 12}, {7, 8}, {11, 12}};
    const int mant7[16] = {0, 1, 2, 3, 4, 8, 16, 32, 64, 96, 112, 120, 124, 126, 127, 85};
    const int mant10[16] = {0, 1, 2, 3, 4, 8, 64, 256, 512, 768, 896, 960, 1008, 1020, 1022, 1023};
    const int smax = bf ? 30 : 24;
    auto pow2neg = [&](int sh, bool neg) -> uint16_t {
      int e = bias - sh;
      uint16_t v = e >= 1 ? uint16_t(e << fbits) : uint16_t(1u << (bias + fbits - 1 - sh));
      return uint16_t(v | (neg ? 0x8000 : 0));
    };
    for (int ki = 0; ki < 12; ki++)
      for (int sh = 0; sh <= smax; sh++)
        for (int neg = 0; neg < 2; neg++) {
          WCase c = zero_case();
          for (int i = 0; i < 16; i++) {
            uint16_t v = uint16_t(((bias) << fbits) | (bf ? mant7[i] : mant10[i]));
            c.A[i * 16 + kc[ki][0]] = v; c.Bt[i * 16 + kc[ki][0]] = v;
            c.A[i * 16 + kc[ki][1]] = uint16_t(bias << fbits);
            c.Bt[i * 16 + kc[ki][1]] = pow2neg(sh, neg);
          }
          s.cs.push_back(c);
        }
  }
  s.layout = "controls(3) F1(single-product) F4(subnormal rows) F5(C-tail window sweep) F3(random 2000)=[sub prefix] F2(two-product) F3(random 8000) F6(mid-step precision, last 12*(smax+1)*2 waves)";
  return s;
}

static WSet make_iu8_set(uint64_t seed) {
  WSet s;
  Rng g(seed);
  { WCase c = zero_case(); for (int i = 0; i < 16; i++) c.A[i * 16 + i] = 1; for (int k = 0; k < 16; k++) for (int j = 0; j < 16; j++) c.Bt[j * 16 + k] = uint16_t((k * 16 + j) & 255); s.cs.push_back(c); }
  { WCase c = zero_case(); for (int i = 0; i < 256; i++) c.C[i] = 0x01000000u + uint32_t(i) * 0x10001u; s.cs.push_back(c); }
  { WCase c = zero_case(); for (int j = 0; j < 16; j++) c.Bt[j * 16 + j] = 1; for (int i = 0; i < 256; i++) c.A[i] = uint16_t(i & 255); s.cs.push_back(c); }
  // corners: uniform byte rows x uniform byte columns x C specials
  static const uint8_t B8[] = {0x00, 0x01, 0x7f, 0x80, 0xff, 0x81, 0x02, 0xfe};
  for (uint8_t a : B8) for (uint8_t b : B8) for (uint32_t cv : I32_S) {
    WCase c = zero_case();
    for (int i = 0; i < 256; i++) { c.A[i] = a; c.Bt[i] = b; c.C[i] = cv; }
    s.cs.push_back(c);
  }
  // ordered ladders: positive partial sums first then negative (saturation-order discriminators)
  for (uint32_t cv : I32_S) for (int pat = 0; pat < 8; pat++) {
    WCase c = zero_case();
    for (int r = 0; r < 16; r++) for (int k = 0; k < 16; k++) {
      int sign = (pat & 1) ? (k >= 8 ? -1 : 1) : (pat & 2) ? (k < 8 ? -1 : 1) : ((k & 1) ? -1 : 1);
      int mag = (pat & 4) ? 127 : 1 + int(g.below(127));
      c.A[r * 16 + k] = uint8_t(mag);
      for (int j = 0; j < 16; j++) c.Bt[j * 16 + k] = uint8_t(sign * (pat & 4 ? 127 : 1 + ((j * 7 + k) % 100)));
    }
    for (int i = 0; i < 256; i++) c.C[i] = cv;
    s.cs.push_back(c);
  }
  s.n_sub = s.cs.size();
  for (int w = 0; w < 6000; w++) {
    WCase c = zero_case();
    int style = int(g.below(4));
    for (int i = 0; i < 256; i++) {
      c.A[i] = style == 0 ? g.u32() & 255 : style == 1 ? B8[g.below(8)] : (g.u32() & 3 ? g.u32() & 255 : 0x80);
      c.Bt[i] = style == 0 ? g.u32() & 255 : style == 2 ? B8[g.below(8)] : (g.below(3) ? g.u32() & 255 : 0x7f);
      c.C[i] = g.below(3) == 0 ? I32_S[g.below(I32_S.size())] : g.below(2) ? g.u32() : uint32_t(int32_t(g.u32()) >> g.below(32));
    }
    s.cs.push_back(c);
  }
  s.layout = "controls(3) corners(uniform bytes) ladders sub-prefix random(6000)";
  return s;
}

static bool np_lite() { const char* e = getenv("NP_LITE"); return e && *e == '1'; }
static void run_wset(int kind, const WSet& s, const char* tag) {
  const bool iu = kind == 3;
  const int nv = iu ? NWV_I : (kind == 0 || kind == 4 ? 8 : NWV_F);
  uint32_t n = s.cs.size();
  std::string inname = fmt("wmma.%s.in", WKIND[kind]);
  write_file(OUT + "/" + inname + ".bin", s.cs.data(), s.cs.size() * sizeof(WCase));
  record(inname, fmt("{\"kind\":\"wmma_inputs\",\"arch\":\"%s\",\"op\":\"%s\",\"cases\":%u,\"n_control\":%u,\"n_sub\":%u,\"layout\":\"%s\",\"record\":\"A u16[256] (r*16+k); Bt u16[256] (c*16+k); C u32[256] (r*16+c); iu8 uses the low byte\",\"%s\":1}", ARCH, WKIND[kind], n, s.n_control, s.n_sub, s.layout.c_str(), tag));
  DevBuf<WCase> din(s.cs);
  const int nvo = wk_out16(kind) ? NVO16 : 8;
  DevBuf<uint32_t> dout(size_t(n) * nvo * 32), mo(1);
  for (int v = 0; v < nv; v++)
    for (int m = 0; m < NMODES; m++) {
      if (v && m) continue;
      if (np_lite() && (v || m)) continue;
      // modifier variants and non-default modes run on the sub-prefix only; V0/default runs the whole set
      uint32_t nn = (v == 0 && m == 0) ? n : s.n_sub;
      wmma_dispatch(kind, v, din, dout, nn, MODES[m], mo);
      HIP(hipGetLastError());
      HIP(hipDeviceSynchronize());
      auto res = dout.get();
      res.resize(size_t(nn) * nvo * 32);
      std::string name = fmt("wmma.%s.v%d.m%d.out", WKIND[kind], v, m);
      write_file(OUT + "/" + name + ".bin", res.data(), res.size() * 4);
      record(name, fmt("{\"kind\":\"wmma_result\",\"arch\":\"%s\",\"op\":\"%s\",\"cases\":%u,\"variant\":%d,\"mods\":\"%s\",\"mode_index\":%d,\"mode_request\":\"0x%08x\",\"mode_readback\":\"0x%08x\",\"inputs\":\"%s\",\"vgprs\":%d,\"record\":\"u32[case][vgpr][lane32]\"}",
                       ARCH, WKIND[kind], nn, v, iu ? WIMODS[v] : WFMODS[v], m, MODES[m], popmode(mo), inname.c_str(), nvo));
    }
}

// NP_KINDS="1,2" restricts the float WMMA kinds (0 bf16->f32, 1 bf16->bf16, 2 f16->f16, 4 f16->f32); NP_LITE=1 runs only the
// base variant at the default MODE (receipts stay small when a focused re-run is needed).
static void suite_wmma() {
  const char* env = getenv("NP_KINDS");
  std::string want = std::string(",") + (env ? env : "0,1,2,4") + ",";
  for (int k : {0, 1, 2, 4})
    if (want.find("," + std::to_string(k) + ",") != std::string::npos) run_wset(k, make_float_set(k, 0x57a000 + k), "float");
}
static void suite_iu8() { run_wset(3, make_iu8_set(0x1a8), "iu8"); }

// ============================================================================================== SWMMAC (gfx12)
// Raw register images, so the index/fragment mapping is discovered rather than assumed.
struct SCase { uint32_t a[4][32]; uint32_t b[8][32]; uint32_t idx[32]; uint32_t c[8][32]; };
// kinds: 0 f32_f16(50) 1 f32_bf16(51) 2 f16_f16(52) 3 bf16_bf16(53) 4 i32_iu8(54) 5 i32_iu4_x32(55) 6 i32_iu4_x64(56)
//        7 fp8_fp8(57) 8 fp8_bf8(58) 9 bf8_fp8(59) 10 bf8_bf8(5a)
static const char* SKIND[] = {"v_swmmac_f32_16x16x32_f16", "v_swmmac_f32_16x16x32_bf16", "v_swmmac_f16_16x16x32_f16", "v_swmmac_bf16_16x16x32_bf16",
                              "v_swmmac_i32_16x16x32_iu8", "v_swmmac_i32_16x16x32_iu4", "v_swmmac_i32_16x16x64_iu4", "v_swmmac_f32_16x16x32_fp8_fp8",
                              "v_swmmac_f32_16x16x32_fp8_bf8", "v_swmmac_f32_16x16x32_bf8_fp8", "v_swmmac_f32_16x16x32_bf8_bf8"};
constexpr int NSK = 11;
// per-kind register counts {A,B,D}
constexpr int SA[NSK] = {4, 4, 4, 4, 2, 1, 2, 2, 2, 2, 2};
constexpr int SB[NSK] = {8, 8, 8, 8, 4, 2, 4, 4, 4, 4, 4};
constexpr int SD[NSK] = {8, 8, 4, 4, 8, 8, 8, 8, 8, 8, 8};
// variants: 0 key0 | 1 key1 | 2 neg_lo:[1,0] | 3 neg_lo:[0,1] | 4 neg_lo:[1,1] / neg_hi:[1,0] (float) | 5 clamp / neg_hi:[0,1] | 6 neg_lo:[1,1] clamp
constexpr int NSV = 7;
#if NP_GFX12
#define SA_(MN, M) asm volatile("s_nop 4\n\t" MN " %[d], %[a], %[b], %[i]" M "\n\ts_nop 7" : [d] "+v"(d) : [a] "v"(a), [b] "v"(b), [i] "v"(ix) : "memory")
#define SFBODY(MN)                                                                                    \
  if constexpr (V == 0) { SA_(MN, ""); }                                                              \
  else if constexpr (V == 1) { SA_(MN, " index_key:1"); }                                             \
  else if constexpr (V == 2) { SA_(MN, " neg_lo:[1,0,0]"); }                                          \
  else if constexpr (V == 3) { SA_(MN, " neg_lo:[0,1,0]"); }                                          \
  else if constexpr (V == 4) { SA_(MN, " neg_hi:[1,0,0]"); }                                          \
  else if constexpr (V == 5) { SA_(MN, " neg_hi:[0,1,0]"); }                                          \
  else if constexpr (V == 6) { SA_(MN, " neg_lo:[1,1,0] neg_hi:[1,1,0]"); }
#define SIBODY(MN)                                                                                    \
  if constexpr (V == 0) { SA_(MN, ""); }                                                              \
  else if constexpr (V == 1) { SA_(MN, " index_key:1"); }                                             \
  else if constexpr (V == 2) { SA_(MN, " neg_lo:[1,0,0]"); }                                          \
  else if constexpr (V == 3) { SA_(MN, " neg_lo:[0,1,0]"); }                                          \
  else if constexpr (V == 4) { SA_(MN, " neg_lo:[1,1,0]"); }                                          \
  else if constexpr (V == 5) { SA_(MN, " clamp"); }                                                   \
  else if constexpr (V == 6) { SA_(MN, " neg_lo:[1,1,0] clamp"); }
// x64 iu4 has no index_key operand: V1 is unused (compiled as the plain form) and is skipped by the host
#define SI64BODY(MN)                                                                                  \
  if constexpr (V == 0 || V == 1) { SA_(MN, ""); }                                                    \
  else if constexpr (V == 2) { SA_(MN, " neg_lo:[1,0,0]"); }                                          \
  else if constexpr (V == 3) { SA_(MN, " neg_lo:[0,1,0]"); }                                          \
  else if constexpr (V == 4) { SA_(MN, " neg_lo:[1,1,0]"); }                                          \
  else if constexpr (V == 5) { SA_(MN, " clamp"); }                                                   \
  else if constexpr (V == 6) { SA_(MN, " neg_lo:[1,1,0] clamp"); }
#define SF8BODY(MN)  /* fp8 forms accept index_key only (no neg_lo/neg_hi) */                          \
  if constexpr (V == 1) { SA_(MN, " index_key:1"); }                                                  \
  else { SA_(MN, ""); }
template <int KIND, int V>
__global__ void __launch_bounds__(32) swmmac_k(const SCase* cs, uint32_t* out, uint32_t n, uint32_t mode, uint32_t* modeout) {
  set_mode(mode);
  const uint32_t w = blockIdx.x, l = threadIdx.x;
  if (w == 0 && l == 0) modeout[0] = get_mode();
  if (w >= n) return;
  const SCase& cc = cs[w];
  uint32_t ix = cc.idx[l];
  constexpr int ND = SD[KIND];
  if constexpr (KIND <= 3) {
    v4i a; v8i b;
    for (int i = 0; i < 4; i++) a[i] = int(cc.a[i][l]);
    for (int i = 0; i < 8; i++) b[i] = int(cc.b[i][l]);
    if constexpr (ND == 8) {
      v8i d; for (int i = 0; i < 8; i++) d[i] = int(cc.c[i][l]);
      if constexpr (KIND == 0) { SFBODY("v_swmmac_f32_16x16x32_f16") } else { SFBODY("v_swmmac_f32_16x16x32_bf16") }
      for (int i = 0; i < 8; i++) out[(size_t(w) * 8 + i) * 32 + l] = uint32_t(d[i]);
    } else {
      v4i d; for (int i = 0; i < 4; i++) d[i] = int(cc.c[i][l]);
      if constexpr (KIND == 2) { SFBODY("v_swmmac_f16_16x16x32_f16") } else { SFBODY("v_swmmac_bf16_16x16x32_bf16") }
      for (int i = 0; i < 4; i++) out[(size_t(w) * 4 + i) * 32 + l] = uint32_t(d[i]);
    }
  } else if constexpr (KIND == 5) {
    int a = int(cc.a[0][l]); v2i b; for (int i = 0; i < 2; i++) b[i] = int(cc.b[i][l]);
    v8i d; for (int i = 0; i < 8; i++) d[i] = int(cc.c[i][l]);
    SIBODY("v_swmmac_i32_16x16x32_iu4")
    for (int i = 0; i < 8; i++) out[(size_t(w) * 8 + i) * 32 + l] = uint32_t(d[i]);
  } else {
    v2i a; v4i b;
    for (int i = 0; i < 2; i++) a[i] = int(cc.a[i][l]);
    for (int i = 0; i < 4; i++) b[i] = int(cc.b[i][l]);
    v8i d; for (int i = 0; i < 8; i++) d[i] = int(cc.c[i][l]);
    if constexpr (KIND == 4) { SIBODY("v_swmmac_i32_16x16x32_iu8") }
    else if constexpr (KIND == 6) { SI64BODY("v_swmmac_i32_16x16x64_iu4") }
    else if constexpr (KIND == 7) { SF8BODY("v_swmmac_f32_16x16x32_fp8_fp8") }
    else if constexpr (KIND == 8) { SF8BODY("v_swmmac_f32_16x16x32_fp8_bf8") }
    else if constexpr (KIND == 9) { SF8BODY("v_swmmac_f32_16x16x32_bf8_fp8") }
    else { SF8BODY("v_swmmac_f32_16x16x32_bf8_bf8") }
    for (int i = 0; i < 8; i++) out[(size_t(w) * 8 + i) * 32 + l] = uint32_t(d[i]);
  }
}
template <int KIND, int N> static void swmmac_launch(int v, DevBuf<SCase>& in, DevBuf<uint32_t>& out, uint32_t n, uint32_t mode, DevBuf<uint32_t>& mo) {
  if constexpr (N > 0) {
    if (v == N - 1) { swmmac_k<KIND, N - 1><<<n, 32>>>(in.p, out.p, n, mode, mo.p); return; }
    swmmac_launch<KIND, N - 1>(v, in, out, n, mode, mo);
  }
}
#define SW_CASE(K) case K: swmmac_launch<K, NSV>(v, in, out, n, mode, mo); break;
static void swmmac_dispatch(int kind, int v, DevBuf<SCase>& in, DevBuf<uint32_t>& out, uint32_t n, uint32_t mode, DevBuf<uint32_t>& mo) {
  switch (kind) { SW_CASE(0) SW_CASE(1) SW_CASE(2) SW_CASE(3) SW_CASE(4) SW_CASE(5) SW_CASE(6) SW_CASE(7) SW_CASE(8) SW_CASE(9) SW_CASE(10) default: break; }
}
#else
static void swmmac_dispatch(int, int, DevBuf<SCase>&, DevBuf<uint32_t>&, uint32_t, uint32_t, DevBuf<uint32_t>&) {}
#endif

static void run_sset(int kind, const std::vector<SCase>& cs, const char* setname, const std::vector<int>& variants, int mode_index, const std::string& note) {
  uint32_t n = cs.size();
  std::string inname = fmt("swmmac.%s.%s.in", SKIND[kind], setname);
  write_file(OUT + "/" + inname + ".bin", cs.data(), cs.size() * sizeof(SCase));
  record(inname, fmt("{\"kind\":\"swmmac_inputs\",\"arch\":\"%s\",\"op\":\"%s\",\"set\":\"%s\",\"cases\":%u,\"note\":\"%s\",\"record\":\"a[4][32] b[8][32] idx[32] c[8][32] u32 raw (reg-major, lane minor)\"}", ARCH, SKIND[kind], setname, n, note.c_str()));
  DevBuf<SCase> din(cs);
  DevBuf<uint32_t> dout(size_t(n) * SD[kind] * 32), mo(1);
  for (int v : variants) {
    swmmac_dispatch(kind, v, din, dout, n, MODES[mode_index], mo);
    HIP(hipGetLastError());
    HIP(hipDeviceSynchronize());
    auto res = dout.get();
    std::string name = fmt("swmmac.%s.%s.v%d.m%d.out", SKIND[kind], setname, v, mode_index);
    write_file(OUT + "/" + name + ".bin", res.data(), res.size() * 4);
    record(name, fmt("{\"kind\":\"swmmac_result\",\"arch\":\"%s\",\"op\":\"%s\",\"set\":\"%s\",\"cases\":%u,\"variant\":%d,\"mode_index\":%d,\"mode_request\":\"0x%08x\",\"mode_readback\":\"0x%08x\",\"inputs\":\"%s\",\"a_regs\":%d,\"b_regs\":%d,\"d_regs\":%d,\"record\":\"u32[case][dreg][lane32]\"}",
                     ARCH, SKIND[kind], setname, n, v, mode_index, MODES[mode_index], popmode(mo), inname.c_str(), SA[kind], SB[kind], SD[kind]));
  }
}

static SCase zero_s() { SCase c; memset(&c, 0, sizeof c); return c; }

// Mapping discovery with v_swmmac_i32_16x16x32_iu8 (exact integer arithmetic, unsigned bytes, C=0).
//  A (compressed): ONE nonzero byte (value 1) per row, at (half hs, byte position ea): lanes of half hs only.
//  B: byte (breg, bbyte) of lane l holds id 1 + 16*(l>>4) + 4*breg + bbyte, identical for every column.
//  => D[row][col] = id of the B element selected by that compressed slot through the index word, for every row at once.
static void discovery() {
  std::vector<SCase> cs;
  Rng g(0x1de);
  std::vector<uint32_t> idxs = {0x00000000, 0x55555555, 0xaaaaaaaa, 0xffffffff, 0x44444444, 0xe4e4e4e4, 0x1b1b1b1b, 0x0000ffff, 0xffff0000};
  for (int i = 0; i < 6; i++) idxs.push_back(g.u32());
  for (uint32_t iw : idxs)
    for (int hs = 0; hs < 2; hs++)
      for (int ea = 0; ea < 8; ea++) {
        SCase c = zero_s();
        for (int l = 0; l < 32; l++) {
          c.idx[l] = iw;
          for (int br = 0; br < 4; br++) {
            uint32_t wv = 0;
            for (int bb = 0; bb < 4; bb++) wv |= uint32_t(1 + 16 * (l >> 4) + 4 * br + bb) << (8 * bb);
            c.b[br][l] = wv;
          }
          if ((l >> 4) == hs) c.a[ea >> 2][l] = 1u << (8 * (ea & 3));
        }
        cs.push_back(c);
      }
  // key variants: V0 = index_key:0, V1 = index_key:1; the per-lane index word may also differ by lane (random per lane)
  run_sset(4, cs, "discover", {0, 1}, 0, "one-hot compressed A byte, B element ids; idx word identical on all lanes");
  cs.clear();
  for (int rep = 0; rep < 64; rep++)
    for (int hs = 0; hs < 2; hs++)
      for (int ea = 0; ea < 8; ea++) {
        SCase c = zero_s();
        for (int l = 0; l < 32; l++) {
          c.idx[l] = g.u32();
          for (int br = 0; br < 4; br++) {
            uint32_t wv = 0;
            for (int bb = 0; bb < 4; bb++) wv |= uint32_t(1 + 16 * (l >> 4) + 4 * br + bb) << (8 * bb);
            c.b[br][l] = wv;
          }
          if ((l >> 4) == hs) c.a[ea >> 2][l] = 1u << (8 * (ea & 3));
        }
        cs.push_back(c);
      }
  run_sset(4, cs, "discover_lane", {0, 1}, 0, "same, index word random per lane (reveals which lane's index word is consumed)");
}

// Sparse logical -> raw packing under hypothesis H (verified by analysis against discover*): see analyze.py `H`.
//   lane l: row = l&15, half h = l>>4. Compressed slot j in [0,16): j = 8h+i, i in [0,8); K group g=j>>1 (K = 4g + pos).
//   A raw element i of lane l is compressed slot 8h+i. Index word of lane l: bits [16*key + 2*i, +2) = pos of slot i.
//   B dense K=32: lane l column l&15; B raw element e (0..15) of lane l is K = 16*h + e.   (to be confirmed by discovery)
// Numerics: each case carries (A_logical[16][32] with 2:4 pattern, B[32][16], C) built by gen; raw bits derive from it.
struct SLogical { uint16_t Av[16][16]; uint8_t pos[16][16]; uint16_t B[32][16]; };

static int elem_bits(int kind) { return kind <= 3 ? 16 : (kind == 4 ? 8 : (kind == 5 || kind == 6 ? 4 : 8)); }
static void put_elem(uint32_t* regs /*[reg][32] flat base for this lane*/, int lane, int e, int bits, uint32_t v) {
  int per = 32 / bits, reg = e / per, sh = (e % per) * bits;
  uint32_t m = (bits == 32 ? 0xffffffffu : ((1u << bits) - 1));
  regs[reg * 32 + lane] = (regs[reg * 32 + lane] & ~(m << sh)) | ((v & m) << sh);
}
// element-granular raw builders for numerics sets (hypothesis H)
static SCase pack_s(int kind, const SLogical& L, const uint32_t* C /*[256] r*16+c, raw acc word*/, int key) {
  SCase c = zero_s();
  const int bits = elem_bits(kind);
  const int nA = kind == 6 ? 32 : kind == 5 ? 8 : (kind <= 3 ? 8 : 8);  // slots per lane-half handled below
  (void)nA;
  const int slots = 8;  // compressed slots per lane (kind<=4,7..10 : 8 slots; iu4 kinds hold more nibbles and use separate generators)
  for (int l = 0; l < 32; l++) {
    int row = l & 15, h = l >> 4;
    uint32_t iw = 0;
    for (int i = 0; i < slots; i++) {
      put_elem(&c.a[0][0], l, i, bits, L.Av[row][8 * h + i]);
      iw |= uint32_t(L.pos[row][8 * h + i] & 3) << (16 * key + 2 * i);
    }
    c.idx[l] = iw;
    for (int e = 0; e < 16 && bits == 16; e++) put_elem(&c.b[0][0], l, e, 16, L.B[16 * h + e][l & 15]);
    for (int e = 0; e < 16 && bits == 8; e++) put_elem(&c.b[0][0], l, e, 8, L.B[16 * h + e][l & 15]);
    for (int j = 0; j < 8; j++) c.c[j][l] = C[(8 * h + j) * 16 + (l & 15)];
  }
  return c;
}

// Numerics: sparse f16/bf16 with the pairing pos = {0,1} for every group (dense-equivalent chain), compared on hardware
// against the dense WMMA of the same 16 products: see analyze.py `swmmac_vs_wmma`.
static void swmmac_numerics() {
  for (int kind = 0; kind <= 3; kind++) {
    const bool bf = kind == 1 || kind == 3;
    const std::vector<uint16_t>& S = bf ? BF16_S : F16_S;
    const int ebits = bf ? 8 : 5, fbits = bf ? 7 : 10, bias = bf ? 127 : 15;
    Rng g(0x5a11 + kind);
    std::vector<SCase> cs;
    // random 2:4 patterns (idx random) + dense-equivalent patterns (pos pairs {0,1}); 6000 + 6000 waves
    for (int w = 0; w < 12000; w++) {
      SLogical L;
      memset(&L, 0, sizeof L);
      const bool dense_eq = w >= 6000;
      int base = bias + int(g.below(bias)) - bias / 2, spread = int(g.below(5)) * 3;
      bool finite_only = g.below(100) < 85;
      auto el = [&]() -> uint16_t {
        uint32_t r = g.below(10);
        uint16_t v = r < 1 ? uint16_t((g.u32() & 1) << 15) : r < 2 ? S[g.below(S.size())] : uint16_t(rnd_float(g, ebits, fbits, base, spread));
        if (finite_only && ((v >> fbits) & ((1 << ebits) - 1)) == (1u << ebits) - 1) v &= ~(1u << fbits);
        return v;
      };
      for (int r = 0; r < 16; r++)
        for (int j = 0; j < 16; j++) {
          L.Av[r][j] = el();
          int p0, p1;
          if (dense_eq) { p0 = 0; p1 = 1; } else { p0 = int(g.below(3)); p1 = p0 + 1 + int(g.below(3 - p0)); }
          L.pos[r][j] = (j & 1) ? p1 : p0;
        }
      for (int k = 0; k < 32; k++) for (int c = 0; c < 16; c++) L.B[k][c] = el();
      uint32_t C[256];
      for (int i = 0; i < 256; i++) {
        if (kind >= 2) C[i] = g.below(8) == 0 ? S[g.below(S.size())] : uint32_t(rnd_float(g, ebits, fbits, base, 3));
        else C[i] = rnd_c(g, 127 + (base - bias) * 2 + int(g.below(15)) - 7, 4);
        if (finite_only && kind < 2 && ((C[i] >> 23) & 255) == 255) C[i] &= ~(1u << 30);
      }
      if (kind >= 2) {
        // 16-bit accumulators are packed two rows per VGPR on gfx12; rebuild in pack below
      }
      SCase sc = pack_s(kind, L, C, 0);
      if (kind >= 2) {
        for (int l = 0; l < 32; l++) { int c = l & 15, h = l >> 4; for (int j = 0; j < 4; j++) sc.c[j][l] = (C[(8 * h + 2 * j) * 16 + c] & 0xffff) | (C[(8 * h + 2 * j + 1) * 16 + c] << 16); for (int j = 4; j < 8; j++) sc.c[j][l] = 0; }
      }
      cs.push_back(sc);
    }
    run_sset(kind, cs, "numerics", {0}, 0, "waves 0..5999 random 2:4 positions; 6000..11999 pos pairs {0,1} (dense-equivalent chain). Packing hypothesis H; see analyze.py");
  }
}


// Exact-value layout validation for every SWMMAC kind: raw register images whose element values are small exactly
// representable numbers (floats) or arbitrary bits (integers), random per-lane index words. analyze.py decodes the images
// with the discovered layout (see swmmac_layout in analyze.py) and requires bit-exact agreement with the exact sum.
static std::vector<uint32_t> swmmac_vals(int kind, bool wide) {
  switch (kind) {
    case 0: case 2: return wide ? std::vector<uint32_t>{0x0000, 0x3c00, 0xbc00, 0x4000, 0xc000, 0x3800, 0xb800, 0x4200, 0xc200, 0x3400, 0x4400}
                                : std::vector<uint32_t>{0x0000, 0x3c00, 0xbc00, 0x4000, 0xc000};
    case 1: case 3: return wide ? std::vector<uint32_t>{0x0000, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x3f00, 0xbf00, 0x4040, 0xc040, 0x3e80, 0x4080}
                                : std::vector<uint32_t>{0x0000, 0x3f80, 0xbf80, 0x4000, 0xc000};
    case 7: return {0x00, 0x38, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x44, 0xc4, 0x28, 0x48};   // e4m3 first operand
    case 8: return {0x00, 0x38, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x44, 0xc4, 0x28, 0x48};
    default: return {};
  }
}
static uint32_t pick_elem(int kind, bool second, Rng& g) {
  // second operand format of the mixed fp8/bf8 kinds: 9 = bf8(A) x fp8(B), 8 = fp8(A) x bf8(B), 7 fp8 x fp8, 10 bf8 x bf8
  if (kind >= 7) {
    static const uint32_t e4m3[] = {0x00, 0x38, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x44, 0xc4, 0x28, 0x48};
    static const uint32_t e5m2[] = {0x00, 0x3c, 0xbc, 0x40, 0xc0, 0x38, 0xb8, 0x42, 0xc2, 0x34, 0x44};
    bool is_e4m3 = (kind == 7) || (kind == 8 && !second) || (kind == 9 && second);
    return is_e4m3 ? e4m3[g.below(11)] : e5m2[g.below(11)];
  }
  if (kind <= 3) { auto v = swmmac_vals(kind, kind <= 1 || true); return v[g.below(v.size())]; }
  return g.u32();  // integer kinds: arbitrary bits
}
static void swmmac_valid() {
  for (int kind = 0; kind < NSK; kind++) {
    Rng g(0xa11d + kind);
    std::vector<SCase> cs;
    const int bitsA = elem_bits(kind);
    const int nsl = (kind == 6) ? 16 : 8;  // compressed slots per lane
    (void)nsl;
    for (int w = 0; w < 1000; w++) {
      SCase c = zero_s();
      for (int l = 0; l < 32; l++) {
        c.idx[l] = g.u32();
        for (int e = 0; e < SA[kind] * (32 / bitsA); e++) put_elem(&c.a[0][0], l, e, bitsA, pick_elem(kind, false, g));
        for (int e = 0; e < SB[kind] * (32 / bitsA); e++) put_elem(&c.b[0][0], l, e, bitsA, pick_elem(kind, true, g));
        for (int j = 0; j < SD[kind]; j++) {
          if (kind >= 4 && kind <= 6) c.c[j][l] = g.below(3) == 0 ? I32_S[g.below(I32_S.size())] : g.below(2) ? g.u32() : uint32_t(int32_t(g.u32()) >> (8 + g.below(24)));
          else if (kind <= 1 || kind >= 7) { float f = float(int(g.below(17)) - 8); memcpy(&c.c[j][l], &f, 4); }
          else { auto v = swmmac_vals(kind, false); c.c[j][l] = v[g.below(v.size())] | (v[g.below(v.size())] << 16); }
        }
      }
      cs.push_back(c);
    }
    std::vector<int> variants;
    if (kind >= 7) variants = {0, 1};
    else if (kind == 6) variants = {0, 2, 3, 4, 5, 6};
    else variants = {0, 1, 2, 3, 4, 5, 6};
    run_sset(kind, cs, "valid", variants, 0, "random raw images, exact small values (float kinds) / arbitrary bits (integer kinds); decode with analyze.py swmmac_layout");
  }
}

static void suite_swmmac_map() {
#if NP_GFX12
  discovery();
#endif
}
static void suite_swmmac_num() {
#if NP_GFX12
  swmmac_numerics();
#endif
}
static void suite_swmmac_valid() {
#if NP_GFX12
  swmmac_valid();
#endif
}

// ============================================================================================== DIV_SCALE (gfx11 + gfx12)
// v_div_scale_f32 D, SDST(wave32 lane mask), S0, S1, S2 over an exponent-field Cartesian set (not a 64-member
// value set): every denominator exponent field e1 (0..255) x every numerator exponent field e2 (0..255) x S1/S2
// mantissa classes x a fixed list of S0 shapes (including S0 == S1 and S0 == S2 bitwise). Records D and the lane's
// SDST bit. Opt-in suite (`div_scale`): not part of `all`, so the older receipts stay reproducible.
__global__ void divscale_k(const uint32_t* in, uint32_t* out, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a = in[3 * size_t(i)], b = in[3 * size_t(i) + 1], c = in[3 * size_t(i) + 2], d, m;
  asm volatile("v_div_scale_f32 %0, %1, %2, %3, %4" : "=v"(d), "=s"(m) : "v"(a), "v"(b), "v"(c));
  out[2 * size_t(i)] = d;
  out[2 * size_t(i) + 1] = (m >> (threadIdx.x & 31)) & 1u;
}
static std::vector<uint32_t> divscale_set() {
  std::vector<uint32_t> v;
  static const uint32_t M1[4] = {0x000000, 0x000001, 0x400000, 0x7fffff};
  static const uint32_t M2[3] = {0x000000, 0x555555, 0x7fffff};
  // S0 shapes: kind 0 = S1 bits, 1 = S2 bits, otherwise a fixed word.
  static const uint32_t S0W[12] = {0, 0, 0x00800000, 0x1fffffff, 0x20000000, 0x2dffffff, 0x2e800000, 0x3fc00000, 0xe4123456, 0x7f7fffff, 0x00000001, 0x80000000};
  static const uint32_t S0S[4] = {0x7fc00000, 0x7f800001, 0x7f800000, 0xff800000};
  for (uint32_t e1 = 0; e1 < 256; e1++)
    for (uint32_t e2 = 0; e2 < 256; e2++)
      for (uint32_t i1 = 0; i1 < 4; i1++)
        for (uint32_t i2 = 0; i2 < 3; i2++) {
          uint32_t sg1 = (e1 + i1 + e2) & 1u, sg2 = (e2 + i2 + (e1 >> 1)) & 1u;
          uint32_t s1 = (sg1 << 31) | (e1 << 23) | M1[i1], s2 = (sg2 << 31) | (e2 << 23) | M2[i2];
          for (int k = 0; k < 12 + 4; k++) {
            uint32_t s0 = k == 0 ? s1 : k == 1 ? s2 : k < 12 ? S0W[k] : S0S[k - 12];
            v.push_back(s0); v.push_back(s1); v.push_back(s2);
          }
        }
  return v;
}
static void suite_div_scale() {
  std::vector<uint32_t> in = divscale_set();
  while ((in.size() / 3) % 256) { in.push_back(0x3f800000); in.push_back(0x3f800000); in.push_back(0x3f800000); }
  uint32_t n = in.size() / 3;
  write_file(OUT + "/div_scale.in.bin", in.data(), in.size() * 4);
  record("div_scale.in", fmt("{\"kind\":\"div_scale_inputs\",\"arch\":\"%s\",\"cases\":%u,\"record\":\"u32 s0,s1,s2\"}", ARCH, n));
  DevBuf<uint32_t> din(in), dout(size_t(2) * n);
  divscale_k<<<n / 256, 256>>>(din.p, dout.p, n);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  auto res = dout.get();
  write_file(OUT + "/div_scale.out.bin", res.data(), res.size() * 4);
  record("div_scale.out", fmt("{\"kind\":\"div_scale_result\",\"arch\":\"%s\",\"cases\":%u,\"record\":\"u32 d, u32 sdst_bit\",\"inputs\":\"div_scale.in\"}", ARCH, n));
}

// ============================================================================================== SCALAR UDIV CHAIN (gfx12)
// The four scalar float instructions of the LLVM AMDGPU udiv expansion, one wave per denominator, stage by stage:
//   x = s_cvt_f32_u32(den); r = v_s_rcp_f32(x); m = s_mul_f32(r, 0x4f7ffffe); u = s_cvt_u32_f32(m)
// for every den in 1..=2^24 (den and the f32 it converts to are exact integers there). Records x, r, m, u.
// Opt-in suite (`scalar_udiv`).
#if NP_GFX12
__global__ void sudiv_k(uint32_t* out, uint32_t base, uint32_t n) {
  uint32_t i = base + blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (i >= n) return;
  uint32_t den = __builtin_amdgcn_readfirstlane(i + 1);
  uint32_t x, r, m, u;
  asm volatile("s_cvt_f32_u32 %0, %1" : "=s"(x) : "s"(den));
  asm volatile("v_s_rcp_f32 %0, %1\n\ts_wait_alu depctr_va_sdst(0)" : "=s"(r) : "s"(x));
  asm volatile("s_mul_f32 %0, %1, 0x4f7ffffe" : "=s"(m) : "s"(r));
  asm volatile("s_cvt_u32_f32 %0, %1" : "=s"(u) : "s"(m));
  if ((threadIdx.x & 31) == 0) {
    out[4 * size_t(i)] = x;
    out[4 * size_t(i) + 1] = r;
    out[4 * size_t(i) + 2] = m;
    out[4 * size_t(i) + 3] = u;
  }
}
static void suite_scalar_udiv() {
  const uint32_t n = 1u << 24;
  DevBuf<uint32_t> dout(size_t(4) * n);
  for (uint32_t base = 0; base < n; base += 1u << 22) {
    sudiv_k<<<((1u << 22) + 7) / 8, 256>>>(dout.p, base, n);
    HIP(hipGetLastError());
  }
  HIP(hipDeviceSynchronize());
  auto res = dout.get();
  write_file(OUT + "/scalar_udiv.out.bin", res.data(), res.size() * 4);
  record("scalar_udiv.out", fmt("{\"kind\":\"scalar_udiv_result\",\"arch\":\"%s\",\"dens\":%u,\"record\":\"u32 x,r,m,u for den = index + 1\"}", ARCH, n));
}
#else
static void suite_scalar_udiv() {}
#endif

// ============================================================================================== MAXMIN_NUM / MINMAX_NUM (gfx12)
// v_maxmin_num_f32 (min_num(max_num(S0,S1),S2)) and v_minmax_num_f32 (max_num(min_num(S0,S1),S2)) over the full Cartesian
// product of a 48-value f32 set (zeros, denormals, normals, infinities, quiet and signalling NaNs with payloads) cubed.
// Opt-in suite (`maxmin`).
#if NP_GFX12
template <int OP> __global__ void maxmin_k(const uint32_t* in, uint32_t* out, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t a = in[3 * size_t(i)], b = in[3 * size_t(i) + 1], c = in[3 * size_t(i) + 2], d;
  if constexpr (OP == 0) asm volatile("v_maxmin_num_f32 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
  else asm volatile("v_minmax_num_f32 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
  out[i] = d;
}
static void suite_maxmin() {
  std::vector<uint32_t> vals = F32_S;
  for (uint32_t x : {0x7fa00000u, 0xffa00001u, 0x7fc00001u, 0xffc12345u, 0x00400000u, 0x80400000u, 0x3f800001u, 0xbf800001u,
                     0x00000003u, 0x80000003u, 0x7f7ffffeu, 0xff7ffffeu, 0x3f000000u, 0xbf000000u, 0x40a00000u, 0xc0a00000u})
    vals.push_back(x);
  std::vector<uint32_t> in;
  for (uint32_t a : vals) for (uint32_t b : vals) for (uint32_t c : vals) { in.push_back(a); in.push_back(b); in.push_back(c); }
  uint32_t n = in.size() / 3, padded = (n + 255) / 256 * 256;
  in.resize(size_t(3) * padded, 0);
  write_file(OUT + "/maxmin.in.bin", in.data(), size_t(3) * n * 4);
  record("maxmin.in", fmt("{\"kind\":\"maxmin_inputs\",\"arch\":\"%s\",\"cases\":%u,\"record\":\"u32 s0,s1,s2\"}", ARCH, n));
  DevBuf<uint32_t> din(in), d0(padded), d1(padded);
  maxmin_k<0><<<padded / 256, 256>>>(din.p, d0.p, n);
  maxmin_k<1><<<padded / 256, 256>>>(din.p, d1.p, n);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  auto r0 = d0.get(), r1 = d1.get();
  write_file(OUT + "/maxmin.maxmin.out.bin", r0.data(), size_t(n) * 4);
  write_file(OUT + "/maxmin.minmax.out.bin", r1.data(), size_t(n) * 4);
  record("maxmin.maxmin.out", fmt("{\"kind\":\"maxmin_result\",\"arch\":\"%s\",\"op\":\"v_maxmin_num_f32\",\"cases\":%u,\"inputs\":\"maxmin.in\"}", ARCH, n));
  record("maxmin.minmax.out", fmt("{\"kind\":\"maxmin_result\",\"arch\":\"%s\",\"op\":\"v_minmax_num_f32\",\"cases\":%u,\"inputs\":\"maxmin.in\"}", ARCH, n));
}
#else
static void suite_maxmin() {}
#endif

// ============================================================================================== identity + main
static void suite_identity() {
  int cnt = 0;
  HIP(hipGetDeviceCount(&cnt));
  hipDeviceProp_t p;
  HIP(hipGetDeviceProperties(&p, 0));
  char bus[64];
  HIP(hipDeviceGetPCIBusId(bus, sizeof bus, 0));
  std::string uuid;
  for (int i = 0; i < 16; i++) uuid += fmt("%02x", (unsigned char)p.uuid.bytes[i]);
  std::string j = fmt("{\"kind\":\"identity\",\"compiled_arch\":\"%s\",\"device_count\":%d,\"gcnArchName\":\"%s\",\"name\":\"%s\",\"pci_bus_id\":\"%s\",\"pciDomainID\":%d,\"pciBusID\":%d,\"pciDeviceID\":%d,\"hip_uuid_hex\":\"%s\",\"warpSize\":%d,\"multiProcessorCount\":%d}",
                      ARCH, cnt, p.gcnArchName, p.name, bus, p.pciDomainID, p.pciBusID, p.pciDeviceID, uuid.c_str(), p.warpSize, p.multiProcessorCount);
  record("identity", j);
  printf("%s\n", j.c_str());
  if (cnt != 1 || strncmp(p.gcnArchName, ARCH, 7) != 0 || p.warpSize != 32) { fprintf(stderr, "wrong device set: need exactly one %s device with warpSize 32\n", ARCH); exit(3); }
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <outdir> <identity|dot|wmma|iu8|swmmac_map|swmmac_num|swmmac_valid|all>[,..]\n", argv[0]); return 2; }
  OUT = argv[1];
  mkdir(OUT.c_str(), 0777);
  std::string suites = std::string(",") + argv[2] + ",";
  auto want = [&](const char* s) { return suites.find(std::string(",") + s + ",") != std::string::npos || suites.find(",all,") != std::string::npos; };
  suite_identity();
  if (want("dot")) suite_dot();
  if (want("iu8")) suite_iu8();
  if (want("wmma")) suite_wmma();
  if (want("swmmac_map")) suite_swmmac_map();
  if (want("swmmac_num")) suite_swmmac_num();
  if (want("swmmac_valid")) suite_swmmac_valid();
  // opt-in only (`all` does not include them)
  auto exact = [&](const char* s) { return suites.find(std::string(",") + s + ",") != std::string::npos; };
  if (exact("div_scale")) suite_div_scale();
  if (exact("scalar_udiv")) suite_scalar_udiv();
  if (exact("maxmin")) suite_maxmin();
  std::string idx;
  for (auto& n : INDEX) idx += n + "\n";
  write_text(OUT + "/INDEX.txt", idx);
  printf("done: %zu records in %s\n", INDEX.size(), OUT.c_str());
  return 0;
}
