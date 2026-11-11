// fp8_probe.hip -- raw gfx1201 hardware capture for the B2 (attention qresident v2) emulator models.
//
// Opcodes (each issued by an explicit inline-asm mnemonic, never a compiler builtin):
//   v_cvt_pk_fp8_f32 (VOP3 0x369, vdst.l / vdst.h select)
//   v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8} (0x46..0x49)
//   global_load_tr_b64 (8-bit transposing load)
// The program records raw operand and result bits only; it contains NO arithmetic model. Every conclusion is drawn
// offline by the Rust replay tests of peacemaker-ir (emu/fp8_receipts.rs, `--ignored`, PM_FP8_RECEIPT_DIR=<outdir>).
//
// Build (CPU only): NP_SRC=fp8_probe.hip build.sh gfx1201 [outdir]
// Run (inside a bounded fabric hold):  fp8_probe <outdir> <suite[,suite...]|all>
//   suites: cvt wmma loadtr
//
// cvt: every one of the 2^32 f32 bit patterns is converted in each of four sweeps (src0=x|src1=1.0 and src0=1.0|src1=x,
//   each for vdst.l and vdst.h, old vdst = 0) at the launch MODE, and the src0=x/vdst.l sweep again at each of the 8
//   MODE values. A sweep is stored as a lossless run-length table of raw 32-bit vdst words: (u32 start, u32 word)
//   pairs sorted by start, a run lasting until the next start (the last until 2^32). A final cross sweep converts a
//   random PAIR (x, y) with a random old vdst and a random half select for all 2^32 x and counts, on the device,
//   the words differing from the composition of the single-operand tables (independence, preserved half).
// wmma: raw input/output cases, see FCase. loadtr: raw per-lane origin-offset planes, see LCase.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1201__)
#else
#error "fp8_probe requires --offload-arch=gfx1201"
#endif
#define HIP(C) do{hipError_t e_=(C);if(e_!=hipSuccess){fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#C,hipGetErrorString(e_));exit(2);}}while(0)

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
constexpr const char* ARCH = "gfx1201";

constexpr uint32_t MODE_KEEP = 0xFFFFFFFFu;
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
static const uint32_t MODES[] = {MODE_KEEP, 0x00, 0xC0, 0x30, 0xF1, 0xF2, 0xF3, 0xFC};
constexpr int NMODES = sizeof(MODES) / sizeof(MODES[0]);

// ------------------------------------------------------------------------------------------------ host utils
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
  bool chance(uint32_t num, uint32_t den) { return below(den) < num; }
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
static void record(const std::string& name, const std::string& json) {
  write_text(OUT + "/" + name + ".json", json);
  INDEX.push_back(name);
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

// ================================================================================================ cvt
// word = v_cvt_pk_fp8_f32 (src0=a, src1=b) into vdst with prior value `old`; SEL 0 = vdst.l, 1 = vdst.h
template <int SEL> __device__ __forceinline__ uint32_t cvt_pk(uint32_t a, uint32_t b, uint32_t old) {
  uint32_t d = old;
  if constexpr (SEL == 0) asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2" : "+v"(d) : "v"(a), "v"(b));
  else asm volatile("v_cvt_pk_fp8_f32 %0.h, %1, %2" : "+v"(d) : "v"(a), "v"(b));
  return d;
}
constexpr uint32_t ONE = 0x3f800000u;
template <int WHICH, int SEL> __device__ __forceinline__ uint32_t cvt_sweep_word(uint32_t x) {
  return WHICH == 0 ? cvt_pk<SEL>(x, ONE, 0u) : cvt_pk<SEL>(ONE, x, 0u);
}
// Appends (x, word) for every x whose word differs from that of x-1 (x == base of the whole sweep always).
template <int WHICH, int SEL>
__global__ void cvt_rle_k(uint64_t base, uint32_t mode, uint32_t* list, uint32_t* count, uint32_t cap, uint32_t* modeout) {
  set_mode(mode);
  uint64_t i = base + uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint32_t x = uint32_t(i);
  if (i == 0 && threadIdx.x == 0) modeout[0] = get_mode();
  uint32_t w = cvt_sweep_word<WHICH, SEL>(x);
  bool boundary = (i == 0);
  if (!boundary) boundary = cvt_sweep_word<WHICH, SEL>(x - 1) != w;
  if (boundary) {
    uint32_t at = atomicAdd(count, 1u);
    if (at < cap) { list[2 * at] = x; list[2 * at + 1] = w; }
  }
}
struct Run { uint32_t start, word; };
template <int WHICH, int SEL>
static std::vector<Run> cvt_sweep(uint32_t mode, uint32_t* modeback) {
  const uint32_t CAP = 1u << 22;
  DevBuf<uint32_t> list(size_t(2) * CAP), cnt(1), mo(1);
  const uint64_t CH = 1ull << 30;
  for (uint64_t base = 0; base < (1ull << 32); base += CH) {
    cvt_rle_k<WHICH, SEL><<<unsigned(CH / 256), 256>>>(base, mode, list.p, cnt.p, CAP, mo.p);
    HIP(hipGetLastError());
    HIP(hipDeviceSynchronize());
  }
  uint32_t n = cnt.get()[0];
  if (n > CAP) { fprintf(stderr, "cvt sweep overflow: %u runs\n", n); exit(4); }
  auto l = list.get();
  std::vector<Run> runs(n);
  for (uint32_t i = 0; i < n; i++) runs[i] = {l[2 * i], l[2 * i + 1]};
  std::sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) { return a.start < b.start; });
  *modeback = popmode(mo);
  return runs;
}
static std::vector<uint32_t> runs_flat(const std::vector<Run>& r) {
  std::vector<uint32_t> v;
  for (auto& x : r) { v.push_back(x.start); v.push_back(x.word); }
  return v;
}
// byte table (start, byte) merged
static std::vector<Run> byte_table(const std::vector<Run>& r, int shift) {
  std::vector<Run> t;
  for (auto& x : r) {
    uint32_t b = (x.word >> shift) & 0xff;
    if (t.empty() || t.back().word != b) t.push_back({x.start, b});
  }
  return t;
}
__device__ __forceinline__ uint32_t tab_lookup(const uint32_t* t, uint32_t n, uint32_t x) {
  uint32_t lo = 0, hi = n;  // find last run with start <= x
  while (hi - lo > 1) {
    uint32_t mid = (lo + hi) >> 1;
    if (t[2 * mid] <= x) lo = mid; else hi = mid;
  }
  return t[2 * lo + 1];
}
__device__ __forceinline__ uint32_t hash32(uint32_t v) {
  v ^= v >> 16; v *= 0x7feb352dU; v ^= v >> 15; v *= 0x846ca68bU; v ^= v >> 16;
  return v;
}
__global__ void cvt_cross_k(uint64_t base, const uint32_t* t0, uint32_t n0, const uint32_t* t1, uint32_t n1, uint32_t* bad, uint32_t* ex, uint32_t mode) {
  set_mode(mode);
  uint32_t x = uint32_t(base + uint64_t(blockIdx.x) * blockDim.x + threadIdx.x);
  uint32_t y = hash32(x ^ 0x9e3779b9u), old = hash32(x * 3u + 1u), r = hash32(x + 0x1234567u);
  // special operand mix: a tenth of the pairs use y == x, another tenth x == 1.0
  if ((r & 15) == 0) y = x;
  uint32_t a = x, b = y;
  if (((r >> 4) & 15) == 0) a = ONE;
  uint32_t ea = tab_lookup(t0, n0, a), eb = tab_lookup(t1, n1, b);
  uint32_t got, want;
  if ((r >> 8) & 1) { got = cvt_pk<1>(a, b, old); want = (old & 0xffffu) | (ea << 16) | (eb << 24); }
  else { got = cvt_pk<0>(a, b, old); want = (old & 0xffff0000u) | ea | (eb << 8); }
  if (got != want) {
    uint32_t at = atomicAdd(bad, 1u);
    if (at < 64) { ex[at * 6 + 0] = a; ex[at * 6 + 1] = b; ex[at * 6 + 2] = old; ex[at * 6 + 3] = (r >> 8) & 1; ex[at * 6 + 4] = got; ex[at * 6 + 5] = want; }
  }
}
static std::string runs_json(const char* what, const std::vector<Run>& r, uint32_t modereq, uint32_t modeback) {
  return fmt("{\"kind\":\"cvt_rle\",\"arch\":\"%s\",\"sweep\":\"%s\",\"runs\":%zu,\"mode_request\":\"0x%08x\",\"mode_readback\":\"0x%08x\",\"record\":\"u32 start,u32 word pairs; run lasts to next start (last to 2^32); word = raw vdst, old vdst = 0\"}",
             ARCH, what, r.size(), modereq, modeback);
}
static void suite_cvt() {
  uint32_t mb;
  struct Sw { const char* name; std::vector<Run> r; } sw[4];
  sw[0] = {"s0_lo", cvt_sweep<0, 0>(MODE_KEEP, &mb)};
  record("cvt.s0_lo", runs_json("s0_lo", sw[0].r, MODE_KEEP, mb));
  write_file(OUT + "/cvt.s0_lo.bin", runs_flat(sw[0].r).data(), sw[0].r.size() * 8);
  sw[1] = {"s1_lo", cvt_sweep<1, 0>(MODE_KEEP, &mb)};
  record("cvt.s1_lo", runs_json("s1_lo", sw[1].r, MODE_KEEP, mb));
  write_file(OUT + "/cvt.s1_lo.bin", runs_flat(sw[1].r).data(), sw[1].r.size() * 8);
  sw[2] = {"s0_hi", cvt_sweep<0, 1>(MODE_KEEP, &mb)};
  record("cvt.s0_hi", runs_json("s0_hi", sw[2].r, MODE_KEEP, mb));
  write_file(OUT + "/cvt.s0_hi.bin", runs_flat(sw[2].r).data(), sw[2].r.size() * 8);
  sw[3] = {"s1_hi", cvt_sweep<1, 1>(MODE_KEEP, &mb)};
  record("cvt.s1_hi", runs_json("s1_hi", sw[3].r, MODE_KEEP, mb));
  write_file(OUT + "/cvt.s1_hi.bin", runs_flat(sw[3].r).data(), sw[3].r.size() * 8);
  for (int m = 1; m < NMODES; m++) {
    auto r = cvt_sweep<0, 0>(MODES[m], &mb);
    std::string nm = fmt("s0_lo.m%d", m);
    record("cvt." + nm, runs_json(nm.c_str(), r, MODES[m], mb));
    write_file(OUT + "/cvt." + nm + ".bin", runs_flat(r).data(), r.size() * 8);
  }
  // single-operand byte tables; the other operand's byte and every other bit must be the constant 1.0 conversion
  auto t0 = byte_table(sw[0].r, 0), t1 = byte_table(sw[1].r, 8);
  uint64_t clean = 0, dirty = 0;
  for (auto& x : sw[0].r) (((x.word >> 8) == 0x38u) ? clean : dirty)++;
  for (auto& x : sw[1].r) ((((x.word & 0xff) == 0x38u) && (x.word >> 16) == 0) ? clean : dirty)++;
  for (auto& x : sw[2].r) ((((x.word >> 24) == 0x38u) && (x.word & 0xffff) == 0) ? clean : dirty)++;
  for (auto& x : sw[3].r) ((((x.word >> 16) & 0xff) == 0x38u && (x.word & 0xffff) == 0) ? clean : dirty)++;
  auto h0 = byte_table(sw[2].r, 16), h1 = byte_table(sw[3].r, 24);
  auto eq = [](const std::vector<Run>& a, const std::vector<Run>& b) { return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const Run& x, const Run& y) { return x.start == y.start && x.word == y.word; }); };
  bool lo_hi_same = eq(t0, h0) && eq(t1, h1), t0_t1_same = eq(t0, t1);
  auto f0 = runs_flat(t0), f1 = runs_flat(t1);
  DevBuf<uint32_t> d0(f0), d1(f1), bad(1), ex(64 * 6);
  uint64_t badtotal = 0;
  for (int rep = 0; rep < 2; rep++) {  // rep 1: same cross sweep at MODE 0xC0 (RNE, denormals preserved) to prove mode independence
    HIP(hipMemset(bad.p, 0, 4));
    uint32_t mode = rep ? 0xC0u : MODE_KEEP;
    const uint64_t CH = 1ull << 30;
    for (uint64_t base = 0; base < (1ull << 32); base += CH) {
      cvt_cross_k<<<unsigned(CH / 256), 256>>>(base, d0.p, uint32_t(t0.size()), d1.p, uint32_t(t1.size()), bad.p, ex.p, mode);
      HIP(hipGetLastError());
      HIP(hipDeviceSynchronize());
    }
    uint32_t nb = bad.get()[0];
    badtotal += nb;
    std::string exs = "[";
    auto e = ex.get();
    for (uint32_t i = 0; i < std::min(nb, 64u); i++)
      exs += fmt("%s[\"0x%08x\",\"0x%08x\",\"0x%08x\",%u,\"0x%08x\",\"0x%08x\"]", i ? "," : "", e[i * 6], e[i * 6 + 1], e[i * 6 + 2], e[i * 6 + 3], e[i * 6 + 4], e[i * 6 + 5]);
    exs += "]";
    record(fmt("cvt.cross.r%d", rep), fmt("{\"kind\":\"cvt_cross\",\"arch\":\"%s\",\"pairs\":4294967296,\"mode_request\":\"0x%08x\",\"mismatches\":%u,\"examples_a_b_old_sel_got_want\":%s,\"note\":\"10%% of pairs y==x, 6%% a==1.0\"}", ARCH, mode, nb, exs.c_str()));
  }
  record("cvt.summary", fmt("{\"kind\":\"cvt_summary\",\"arch\":\"%s\",\"runs_s0_lo\":%zu,\"runs_s1_lo\":%zu,\"byte_runs_t0\":%zu,\"byte_runs_t1\":%zu,\"clean_runs\":%llu,\"dirty_runs\":%llu,\"lo_hi_tables_equal\":%d,\"s0_s1_tables_equal\":%d,\"cross_mismatches_total\":%llu}",
                            ARCH, sw[0].r.size(), sw[1].r.size(), t0.size(), t1.size(), (unsigned long long)clean, (unsigned long long)dirty, lo_hi_same, t0_t1_same, (unsigned long long)badtotal));
  printf("cvt: runs %zu/%zu clean=%llu dirty=%llu lo_hi_same=%d s0_s1_same=%d cross_bad=%llu\n", sw[0].r.size(), sw[1].r.size(), (unsigned long long)clean, (unsigned long long)dirty, lo_hi_same, t0_t1_same, (unsigned long long)badtotal);
}

// ================================================================================================ wmma
// kinds: 0 fp8_fp8 (0x46), 1 fp8_bf8, 2 bf8_fp8, 3 bf8_bf8 ; A = first format, B = second
struct FCase { uint8_t A[256]; uint8_t Bt[256]; uint32_t C[256]; };  // A[r*16+k], Bt[c*16+k] (physical k), C[r*16+c]
static const char* FKIND[] = {"v_wmma_f32_16x16x16_fp8_fp8", "v_wmma_f32_16x16x16_fp8_bf8", "v_wmma_f32_16x16x16_bf8_fp8", "v_wmma_f32_16x16x16_bf8_bf8"};
constexpr int NFV = 4;
static const char* FMODS[] = {"", "neg_lo:[0,0,1]", "neg_hi:[0,0,1]", "neg_lo:[0,0,1] neg_hi:[0,0,1]"};
#define WA(MN, M) asm volatile("s_nop 4\n\t" MN " %[d], %[a], %[b], %[d]" M "\n\ts_nop 7" : [d] "+v"(d) : [a] "v"(a), [b] "v"(b) : "memory")
#define WFB(MN)                                                       \
  if constexpr (V == 0) { WA(MN, ""); }                               \
  else if constexpr (V == 1) { WA(MN, " neg_lo:[0,0,1]"); }           \
  else if constexpr (V == 2) { WA(MN, " neg_hi:[0,0,1]"); }           \
  else { WA(MN, " neg_lo:[0,0,1] neg_hi:[0,0,1]"); }
template <int KIND, int V>
__global__ void __launch_bounds__(32) fwmma_k(const FCase* cs, uint32_t* out, uint32_t n, uint32_t mode, uint32_t* modeout) {
  set_mode(mode);
  const uint32_t w = blockIdx.x, l = threadIdx.x, r = l & 15, h = l >> 4;
  if (w == 0 && l == 0) modeout[0] = get_mode();
  if (w >= n) return;
  const FCase& cc = cs[w];
  v2i a, b;
  for (int i = 0; i < 2; i++) {
    int k0 = int(8 * h) + 4 * i;
    a[i] = int(uint32_t(cc.A[r * 16 + k0]) | (uint32_t(cc.A[r * 16 + k0 + 1]) << 8) | (uint32_t(cc.A[r * 16 + k0 + 2]) << 16) | (uint32_t(cc.A[r * 16 + k0 + 3]) << 24));
    b[i] = int(uint32_t(cc.Bt[r * 16 + k0]) | (uint32_t(cc.Bt[r * 16 + k0 + 1]) << 8) | (uint32_t(cc.Bt[r * 16 + k0 + 2]) << 16) | (uint32_t(cc.Bt[r * 16 + k0 + 3]) << 24));
  }
  v8i d;
  for (int j = 0; j < 8; j++) d[j] = int(cc.C[(8 * h + j) * 16 + r]);
  if constexpr (KIND == 0) { WFB("v_wmma_f32_16x16x16_fp8_fp8") }
  else if constexpr (KIND == 1) { WFB("v_wmma_f32_16x16x16_fp8_bf8") }
  else if constexpr (KIND == 2) { WFB("v_wmma_f32_16x16x16_bf8_fp8") }
  else { WFB("v_wmma_f32_16x16x16_bf8_bf8") }
  for (int j = 0; j < 8; j++) out[(size_t(w) * 8 + j) * 32 + l] = uint32_t(d[j]);
}
template <int KIND, int N> static void fwmma_launch(int v, DevBuf<FCase>& in, DevBuf<uint32_t>& out, uint32_t n, uint32_t mode, DevBuf<uint32_t>& mo) {
  if constexpr (N > 0) {
    if (v == N - 1) { fwmma_k<KIND, N - 1><<<n, 32>>>(in.p, out.p, n, mode, mo.p); return; }
    fwmma_launch<KIND, N - 1>(v, in, out, n, mode, mo);
  }
}
static void fwmma_dispatch(int kind, int v, DevBuf<FCase>& in, DevBuf<uint32_t>& out, uint32_t n, uint32_t mode, DevBuf<uint32_t>& mo) {
  switch (kind) {
    case 0: fwmma_launch<0, NFV>(v, in, out, n, mode, mo); break;
    case 1: fwmma_launch<1, NFV>(v, in, out, n, mode, mo); break;
    case 2: fwmma_launch<2, NFV>(v, in, out, n, mode, mo); break;
    default: fwmma_launch<3, NFV>(v, in, out, n, mode, mo); break;
  }
}

// 8-bit float formats. fp8 = E4M3 (bias 7, only NaN = S.1111.111, no infinity), bf8 = E5M2 (bias 15, IEEE-style).
struct F8 { int ebits, mbits, bias; bool inf; };
static const F8 FP8 = {4, 3, 7, false}, BF8 = {5, 2, 15, true};
static uint8_t f8(const F8& f, int s, int e, int m) { return uint8_t((s << 7) | (e << f.mbits) | m); }
static bool f8_isnan(const F8& f, uint8_t c) {
  int e = (c >> f.mbits) & ((1 << f.ebits) - 1), m = c & ((1 << f.mbits) - 1);
  return f.inf ? (e == (1 << f.ebits) - 1 && m != 0) : (e == (1 << f.ebits) - 1 && m == (1 << f.mbits) - 1);
}
static bool f8_isinf(const F8& f, uint8_t c) {
  int e = (c >> f.mbits) & ((1 << f.ebits) - 1), m = c & ((1 << f.mbits) - 1);
  return f.inf && e == (1 << f.ebits) - 1 && m == 0;
}
struct VecParams {
  int center, spread;        // exponent field centre and half-width (clamped to the finite range)
  uint32_t zero_pm, sub_pm, special_pm;  // per-mille zero / subnormal / (nan|inf) probabilities per element
};
static uint8_t gen_elem(Rng& g, const F8& f, const VecParams& p) {
  const int emax = (1 << f.ebits) - 1;
  int s = g.u32() & 1;
  uint32_t r = g.below(1000);
  if (r < p.zero_pm) return f8(f, s, 0, 0);
  r -= p.zero_pm;
  if (r < p.sub_pm) return f8(f, s, 0, 1 + g.below((1 << f.mbits) - 1));
  r -= p.sub_pm;
  if (r < p.special_pm) {
    if (f.inf) return g.chance(1, 2) ? f8(f, s, emax, 0) : f8(f, s, emax, 1 + g.below((1 << f.mbits) - 1));
    return f8(f, s, emax, (1 << f.mbits) - 1);
  }
  int top = f.inf ? emax - 1 : emax;  // largest finite exponent field (E4M3: 15 with m<7)
  int e = p.center + int(g.below(2 * p.spread + 1)) - p.spread;
  e = std::max(1, std::min(top, e));
  int m = g.below(1 << f.mbits);
  if (!f.inf && e == emax && m == (1 << f.mbits) - 1) m--;
  return f8(f, s, e, m);
}
static uint32_t gen_c(Rng& g, int center_pe /*unbiased exponent of a typical product*/) {
  uint32_t r = g.below(100);
  if (r < 12) return g.chance(1, 2) ? 0u : 0x80000000u;
  if (r < 17) return (g.u32() & 0x80000000u) | (g.u32() & 0x7fffffu);  // subnormal
  if (r < 18) { static const uint32_t sp[] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc01234u, 0x7f800001u}; return sp[g.below(5)]; }
  int e = center_pe + int(g.below(81)) - 40;
  e = std::max(-126, std::min(127, e));
  uint32_t frac = g.u32() & 0x7fffffu;
  uint32_t t = g.below(6);
  if (t == 0) frac = 0;
  else if (t == 1) frac &= 0x7f0000u;
  else if (t == 2) frac |= 0x1fu;
  return (g.u32() & 0x80000000u) | (uint32_t(e + 127) << 23) | frac;
}

// exact value of an 8-bit float code as mant * 2^exp (false for NaN/inf)
static bool f8_val(const F8& f, uint8_t c, long long& mant, int& ex) {
  if (f8_isnan(f, c) || f8_isinf(f, c)) return false;
  int e = (c >> f.mbits) & ((1 << f.ebits) - 1), m = c & ((1 << f.mbits) - 1);
  mant = e ? (m | (1 << f.mbits)) : m;
  ex = std::max(e, 1) - f.bias - f.mbits;
  if (c & 0x80) mant = -mant;
  return true;
}
// s * 2^-40 -> f32 bits, RNE (|value| in [2^-32, 2^40): always a normal f32)
static uint32_t scaled_to_f32(__int128 s) {
  if (s == 0) return 0;
  uint32_t sign = s < 0 ? 0x80000000u : 0;
  unsigned __int128 m = s < 0 ? (unsigned __int128)(-s) : (unsigned __int128)s;
  int top = 127;
  while (!((m >> top) & 1)) top--;
  int e = top - 40;
  unsigned __int128 sig;
  if (top > 23) {
    int sh = top - 23;
    sig = m >> sh;
    unsigned __int128 rem = m & ((((unsigned __int128)1) << sh) - 1), half = ((unsigned __int128)1) << (sh - 1);
    if (rem > half || (rem == half && (sig & 1))) sig++;
    if (sig == (1u << 24)) { sig >>= 1; e++; }
  } else {
    sig = m << (23 - top);
  }
  return sign | (uint32_t(e + 127) << 23) | uint32_t(sig & 0x7fffff);
}
static void gen_case(Rng& g, int kind, FCase& c) {
  const F8 fa = (kind == 0 || kind == 1) ? FP8 : BF8, fb = (kind == 0 || kind == 2) ? FP8 : BF8;
  memset(&c, 0, sizeof c);
  uint32_t fam = g.below(100);
  VecParams pa, pb;
  auto cen = [&](const F8& f) { return 1 + int(g.below((1 << f.ebits) - 2)); };
  pa = {cen(fa), (int)g.below(4), 0, 20, 4};
  pb = {cen(fb), (int)g.below(4), 0, 20, 4};
  if (g.chance(1, 4)) { pa.spread = 1 + g.below(8); pb.spread = 1 + g.below(8); }
  if (g.chance(1, 6)) { pa.sub_pm = 400; pb.sub_pm = 400; }
  if (g.chance(1, 8)) { pa.zero_pm = 500; pb.zero_pm = 500; }
  if (g.chance(1, 40)) { pa.special_pm = 40; pb.special_pm = 40; }
  else if (g.chance(1, 3)) { pa.special_pm = 0; pb.special_pm = 0; }
  // support set: positions outside are zero for both operands (so a product needs both operands set)
  bool sparse = fam < 55;
  bool support[16];
  int ns = 16;
  if (sparse) {
    ns = 1 + g.below(g.chance(2, 3) ? 4 : 10);
    for (int k = 0; k < 16; k++) support[k] = false;
    for (int n = 0; n < ns;) { int k = g.below(16); if (!support[k]) { support[k] = true; n++; } }
  } else {
    for (int k = 0; k < 16; k++) support[k] = true;
  }
  // mirror pairs: a equal, b sign-flipped -> cancelling products
  int mir[4][2], nm = 0;
  if (g.chance(1, 3)) {
    nm = 1 + g.below(3);
    for (int i = 0; i < nm; i++) { mir[i][0] = g.below(16); mir[i][1] = g.below(16); if (mir[i][0] == mir[i][1]) mir[i][1] ^= 1; }
  }
  bool near = g.chance(1, 3);
  for (int r = 0; r < 16; r++) {
    for (int k = 0; k < 16; k++) c.A[r * 16 + k] = support[k] ? gen_elem(g, fa, pa) : f8(fa, g.u32() & 1, 0, 0);
    for (int i = 0; i < nm; i++) {
      uint8_t v = c.A[r * 16 + mir[i][0]];
      if (near && g.chance(1, 2)) v ^= (1 << g.below(fa.mbits));
      c.A[r * 16 + mir[i][1]] = v;
    }
  }
  for (int col = 0; col < 16; col++) {
    for (int k = 0; k < 16; k++) c.Bt[col * 16 + k] = support[k] ? gen_elem(g, fb, pb) : f8(fb, g.u32() & 1, 0, 0);
    for (int i = 0; i < nm; i++) c.Bt[col * 16 + mir[i][1]] = c.Bt[col * 16 + mir[i][0]] ^ 0x80;
  }
  int pe = (pa.center - fa.bias) + (pb.center - fb.bias);
  for (int i = 0; i < 256; i++) c.C[i] = gen_c(g, pe);
  // cancellation family: C := nearest f32 of -(exact dot product), optionally +-1 f32 unit / x2 / half
  if (g.chance(1, 3)) {
    for (int r = 0; r < 16; r++)
      for (int col = 0; col < 16; col++) {
        if (!g.chance(2, 3)) continue;
        __int128 S = 0;
        bool ok = true;
        for (int k = 0; k < 16 && ok; k++) {
          long long ma, mb; int ea, eb;
          if (!f8_val(fa, c.A[r * 16 + k], ma, ea) || !f8_val(fb, c.Bt[col * 16 + k], mb, eb)) { ok = false; break; }
          S += (__int128)(ma * mb) * ((__int128)1 << (ea + eb + 40));
        }
        if (!ok) continue;
        uint32_t cb = scaled_to_f32(-S);
        uint32_t t = g.below(8);
        if (t == 1) cb += 1; else if (t == 2 && (cb & 0x7fffffff)) cb -= 1;
        else if (t == 3) cb ^= 0x80000000u;
        else if (t == 4 && (cb & 0x7fffffff)) cb = (cb & 0x80000000u) | ((cb & 0x7fffffff) + 0x00800000u);
        else if (t == 5 && (cb & 0x7fffffff) > 0x00800000u) cb = (cb & 0x80000000u) | ((cb & 0x7fffffff) - 0x00800000u);
        c.C[r * 16 + col] = cb;
      }
  }
}
static FCase zero_fcase() { FCase c; memset(&c, 0, sizeof c); return c; }
static std::vector<FCase> make_fset(int kind, uint32_t nrand, uint64_t seed) {
  const F8 fa = (kind == 0 || kind == 1) ? FP8 : BF8, fb = (kind == 0 || kind == 2) ? FP8 : BF8;
  std::vector<FCase> cs;
  // ---- controls: A[r][k] = 1.0 at k == r ; Bt[c][k] distinct codes -> D[r][c] = C + Bt[c][r]
  {
    FCase c = zero_fcase();
    for (int r = 0; r < 16; r++) c.A[r * 16 + r] = f8(fa, 0, fa.bias, 0);
    for (int col = 0; col < 16; col++)
      for (int k = 0; k < 16; k++) c.Bt[col * 16 + k] = f8(fb, (col >> 2) & 1, fb.bias - 2 + (k & 3), (col * 5 + k * 3) & ((1 << fb.mbits) - 1));
    cs.push_back(c);
    // transposed role: B one-hot, A distinct
    FCase d = zero_fcase();
    for (int col = 0; col < 16; col++) d.Bt[col * 16 + col] = f8(fb, 0, fb.bias, 0);
    for (int r = 0; r < 16; r++)
      for (int k = 0; k < 16; k++) d.A[r * 16 + k] = f8(fa, (r >> 2) & 1, fa.bias - 2 + (k & 3), (r * 5 + k * 3) & ((1 << fa.mbits) - 1));
    cs.push_back(d);
    // C passthrough
    FCase e = zero_fcase();
    for (int i = 0; i < 256; i++) e.C[i] = 0x3f800000u + i * 0x00011111u;
    cs.push_back(e);
  }
  // ---- exhaustive single-product Cartesian: every (a code, b code) at all 16 k positions (diagonal), C classes cycled
  {
    static const uint32_t CS[] = {0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x35800000u, 0xb5800001u, 0x4b7fffffu, 0xcb000001u,
                                  0x00000001u, 0x80000001u, 0x7f7fffffu, 0x00800000u, 0x3dcccccdu, 0xc2f6e979u, 0x33800000u, 0x7fc00000u};
    for (uint32_t a = 0; a < 256; a++)
      for (uint32_t b = 0; b < 256; b++) {
        FCase c = zero_fcase();
        for (int i = 0; i < 16; i++) { c.A[i * 16 + i] = uint8_t(a); c.Bt[i * 16 + i] = uint8_t(b); }
        for (int r = 0; r < 16; r++)
          for (int col = 0; col < 16; col++) c.C[r * 16 + col] = CS[(r + col + a * 7 + b * 3) & 15];
        cs.push_back(c);
      }
  }
  uint32_t nsub = uint32_t(cs.size());
  (void)nsub;
  Rng g(seed);
  for (uint32_t i = 0; i < nrand; i++) { FCase c; gen_case(g, kind, c); cs.push_back(c); }
  return cs;
}
static void run_fset(int kind, uint32_t nrand, uint64_t seed) {
  auto cs = make_fset(kind, nrand, seed);
  uint32_t n = uint32_t(cs.size());
  uint32_t nsub = std::min<uint32_t>(n, 3 + 65536 + 20000);  // controls + single-product Cartesian + a random prefix: variants / modes
  std::string inname = fmt("fwmma.%s.in", FKIND[kind]);
  write_file(OUT + "/" + inname + ".bin", cs.data(), cs.size() * sizeof(FCase));
  record(inname, fmt("{\"kind\":\"fwmma_inputs\",\"arch\":\"%s\",\"op\":\"%s\",\"cases\":%u,\"n_control\":3,\"n_single_cartesian\":65536,\"n_random\":%u,\"seed\":%llu,\"record\":\"A u8[256] (r*16+k, physical k = 8*(lane>>4)+byte); Bt u8[256] (c*16+k); C u32[256] (r*16+c); D VGPR j lane l = row 8*(l>>4)+j, col l&15\"}", ARCH, FKIND[kind], n, nrand, (unsigned long long)seed));
  DevBuf<FCase> din(cs);
  DevBuf<uint32_t> dout(size_t(n) * 8 * 32), mo(1);
  for (int v = 0; v < NFV; v++)
    for (int m = 0; m < NMODES; m++) {
      if (v && m) continue;
      if (kind != 0 && m) continue;  // MODE sweep on fp8_fp8 only; modifier variants on every kind
      uint32_t nn = (v == 0 && m == 0) ? n : nsub;
      fwmma_dispatch(kind, v, din, dout, nn, MODES[m], mo);
      HIP(hipGetLastError());
      HIP(hipDeviceSynchronize());
      auto res = dout.get();
      res.resize(size_t(nn) * 8 * 32);
      std::string name = fmt("fwmma.%s.v%d.m%d.out", FKIND[kind], v, m);
      write_file(OUT + "/" + name + ".bin", res.data(), res.size() * 4);
      record(name, fmt("{\"kind\":\"fwmma_result\",\"arch\":\"%s\",\"op\":\"%s\",\"cases\":%u,\"variant\":%d,\"mods\":\"%s\",\"mode_index\":%d,\"mode_request\":\"0x%08x\",\"mode_readback\":\"0x%08x\",\"inputs\":\"%s\",\"record\":\"u32[case][vgpr 8][lane 32]\"}", ARCH, FKIND[kind], nn, v, FMODS[v], m, MODES[m], popmode(mo), inname.c_str()));
    }
}
static void suite_wmma() {
  const char* e = getenv("FP8_NRAND");
  uint32_t nr = e ? uint32_t(atoi(e)) : 400000;
  const char* se = getenv("FP8_SEED");
  uint64_t sd = se ? strtoull(se, nullptr, 0) : 0xf8a000;
  run_fset(0, nr, sd + 0);
  run_fset(1, nr / 3, sd + 1);
  run_fset(2, nr / 3, sd + 2);
  run_fset(3, nr / 3, sd + 3);
}

// ================================================================================================ load_tr
// global_load_tr_b64 reads 8 bytes per lane from that lane's address and transposes across lanes. Three byte planes
// (byte at offset o = (o >> 8*p) & 255 for p = 0,1,2) recover the 24-bit source offset of every result byte.
constexpr uint32_t LT_SPAN = 1u << 24;
struct LCase { uint32_t mask; uint32_t off[32]; };
__device__ __forceinline__ v2i ld_tr(const unsigned char* p, v2i prev) {
  asm volatile("global_load_tr_b64 %0, %1, off\n\ts_wait_loadcnt 0" : "+v"(prev) : "v"(p) : "memory");
  return prev;
}
__device__ __forceinline__ v2i ld_tr_s(const unsigned char* sbase, uint32_t voff, v2i prev) {
  asm volatile("global_load_tr_b64 %0, %1, %2 offset:24\n\ts_wait_loadcnt 0" : "+v"(prev) : "v"(voff), "s"(sbase) : "memory");
  return prev;
}
template <int FORM>
__global__ void __launch_bounds__(32) ltr_k(const LCase* cs, const unsigned char* p0, const unsigned char* p1, const unsigned char* p2, uint32_t* out, uint32_t n) {
  const uint32_t w = blockIdx.x, l = threadIdx.x;
  if (w >= n) return;
  const LCase& c = cs[w];
  uint32_t off = c.off[l];
  v2i r0 = {int(0xA5A5A5A5u), int(0x5A5A5A5Au)}, r1 = r0, r2 = r0;
  // Every lane (also an EXEC-inactive one) holds a valid address register: the instruction may read inactive lanes' VADDR.
  const unsigned char* q0 = p0 + off;
  const unsigned char* q1 = p1 + off;
  const unsigned char* q2 = p2 + off;
  uint32_t vo = off - 24;
  asm volatile("" : "+v"(q0), "+v"(q1), "+v"(q2), "+v"(vo));
  if ((c.mask >> l) & 1) {
    if constexpr (FORM == 0) { r0 = ld_tr(q0, r0); r1 = ld_tr(q1, r1); r2 = ld_tr(q2, r2); }
    else { r0 = ld_tr_s(p0, vo, r0); r1 = ld_tr_s(p1, vo, r1); r2 = ld_tr_s(p2, vo, r2); }
  }
  uint32_t* o = out + (size_t(w) * 3 * 32 + l) * 2;
  o[0] = uint32_t(r0.x); o[1] = uint32_t(r0.y);
  o[64] = uint32_t(r1.x); o[65] = uint32_t(r1.y);
  o[128] = uint32_t(r2.x); o[129] = uint32_t(r2.y);
}
static const uint32_t A3B_STRIDE = 516, DENSE_STRIDE = 1032;
static void run_lset(const char* name, int form, const std::vector<LCase>& cs, DevBuf<unsigned char>& p0, DevBuf<unsigned char>& p1, DevBuf<unsigned char>& p2) {
  uint32_t n = uint32_t(cs.size());
  write_file(OUT + "/ltr." + name + ".in.bin", cs.data(), cs.size() * sizeof(LCase));
  DevBuf<LCase> din(cs);
  DevBuf<uint32_t> dout(size_t(n) * 3 * 32 * 2);
  if (form == 0) ltr_k<0><<<n, 32>>>(din.p, p0.p, p1.p, p2.p, dout.p, n);
  else ltr_k<1><<<n, 32>>>(din.p, p0.p, p1.p, p2.p, dout.p, n);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  auto res = dout.get();
  write_file(OUT + "/ltr." + name + ".out.bin", res.data(), res.size() * 4);
  record(fmt("ltr.%s", name), fmt("{\"kind\":\"ltr\",\"arch\":\"%s\",\"set\":\"%s\",\"form\":\"%s\",\"cases\":%u,\"span\":%u,\"record_in\":\"u32 mask, u32 off[32] (byte offsets into every plane)\",\"record_out\":\"u32[case][plane 3][lane 32][2]; sentinel 0xA5A5A5A5/0x5A5A5A5A when the lane is inactive; plane p byte = (offset >> 8p) & 255\",\"saddr_form_note\":\"form 1 issues global_load_tr_b64 vdst, v_off, s[plane] offset:24 with v_off = off-24\"}", ARCH, name, form == 0 ? "vaddr64" : "saddr+voff32+imm24", n, LT_SPAN));
}
static void suite_loadtr() {
  std::vector<unsigned char> h(LT_SPAN);
  DevBuf<unsigned char> p0(LT_SPAN), p1(LT_SPAN), p2(LT_SPAN);
  for (int p = 0; p < 3; p++) {
    for (uint32_t o = 0; o < LT_SPAN; o++) h[o] = (o >> (8 * p)) & 255;
    unsigned char* dst = p == 0 ? p0.p : p == 1 ? p1.p : p2.p;
    HIP(hipMemcpy(dst, h.data(), LT_SPAN, hipMemcpyHostToDevice));
  }
  Rng g(0x17f0a);
  auto rnd_off = [&](uint32_t align) { return (g.below(LT_SPAN - 4096) & ~(align - 1)) + 64; };
  std::vector<LCase> cs;
  // T0 contiguous 64 B apart (and 8 B apart), then identical addresses
  { LCase c; c.mask = 0xffffffffu; for (int l = 0; l < 32; l++) c.off[l] = 4096 + 64 * l; cs.push_back(c); }
  { LCase c; c.mask = 0xffffffffu; for (int l = 0; l < 32; l++) c.off[l] = 8192 + 8 * l; cs.push_back(c); }
  { LCase c; c.mask = 0xffffffffu; for (int l = 0; l < 32; l++) c.off[l] = 16384; cs.push_back(c); }
  { LCase c; c.mask = 0xffffffffu; for (int l = 0; l < 32; l++) c.off[l] = 20000 + 1032 * (l & 7) + 16 * (l >> 3); cs.push_back(c); }
  run_lset("seq", 0, cs, p0, p1, p2);
  // T1 random 8-aligned / 4-aligned / byte-aligned, per-lane independent
  for (int al : {8, 4, 2, 1}) {
    cs.clear();
    for (int i = 0; i < 40000; i++) { LCase c; c.mask = 0xffffffffu; for (int l = 0; l < 32; l++) c.off[l] = rnd_off(al); cs.push_back(c); }
    run_lset(fmt("rand_a%d", al).c_str(), 0, cs, p0, p1, p2);
  }
  // T2 the qresident v2 address pattern for both strides: lane address from the kernel's skey / half terms
  for (uint32_t stride : {A3B_STRIDE, DENSE_STRIDE}) {
    cs.clear();
    const uint32_t kvh_n = stride == A3B_STRIDE ? 2 : 4;
    for (int i = 0; i < 40000; i++) {
      LCase c; c.mask = 0xffffffffu;
      uint32_t kt = 64 * g.below(16), sub = g.below(4), kvh = g.below(kvh_n), dc = g.below(16), seq = 1 + g.below(1024);
      for (int l = 0; l < 32; l++) {
        uint32_t skey = 8 * (l >> 4) + 4 * ((l >> 3) & 1) + (l & 3);
        uint32_t gk = std::min(kt + sub * 16 + skey, seq - 1);
        c.off[l] = gk * stride + kvh * 256 + dc * 16 + 8 * ((l >> 2) & 1) + 64;
      }
      cs.push_back(c);
    }
    run_lset(fmt("qres_s%u", stride).c_str(), 0, cs, p0, p1, p2);
  }
  // T3 partial exec
  cs.clear();
  for (int i = 0; i < 40000; i++) { LCase c; c.mask = g.u32(); if (i & 1) c.mask &= g.u32(); for (int l = 0; l < 32; l++) c.off[l] = rnd_off(8); cs.push_back(c); }
  run_lset("exec8", 0, cs, p0, p1, p2);
  // T4 saddr form (offset:24)
  for (int al : {8, 4, 1}) {
    cs.clear();
    for (int i = 0; i < 20000; i++) { LCase c; c.mask = 0xffffffffu; for (int l = 0; l < 32; l++) c.off[l] = rnd_off(al) + 24; cs.push_back(c); }
    run_lset(fmt("saddr_a%d", al).c_str(), 1, cs, p0, p1, p2);
  }
}

static void suite_identity() {
  int cnt = 0;
  HIP(hipGetDeviceCount(&cnt));
  hipDeviceProp_t p;
  HIP(hipGetDeviceProperties(&p, 0));
  char bus[64];
  HIP(hipDeviceGetPCIBusId(bus, sizeof bus, 0));
  std::string uuid;
  for (int i = 0; i < 16; i++) uuid += fmt("%02x", (unsigned char)p.uuid.bytes[i]);
  std::string j = fmt("{\"kind\":\"identity\",\"compiled_arch\":\"%s\",\"device_count\":%d,\"gcnArchName\":\"%s\",\"name\":\"%s\",\"pci_bus_id\":\"%s\",\"hip_uuid_hex\":\"%s\",\"warpSize\":%d}", ARCH, cnt, p.gcnArchName, p.name, bus, uuid.c_str(), p.warpSize);
  record("identity", j);
  printf("%s\n", j.c_str());
  if (cnt != 1 || strncmp(p.gcnArchName, ARCH, 7) != 0 || p.warpSize != 32) { fprintf(stderr, "wrong device set: need exactly one %s device with warpSize 32\n", ARCH); exit(3); }
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <outdir> <cvt|wmma|loadtr|all>[,..]\n", argv[0]); return 2; }
  OUT = argv[1];
  mkdir(OUT.c_str(), 0777);
  std::string suites = std::string(",") + argv[2] + ",";
  auto want = [&](const char* s) { return suites.find(std::string(",") + s + ",") != std::string::npos || suites.find(",all,") != std::string::npos; };
  suite_identity();
  if (want("cvt")) suite_cvt();
  if (want("loadtr")) suite_loadtr();
  if (want("wmma")) suite_wmma();
  std::string idx;
  for (auto& n : INDEX) idx += n + "\n";
  write_text(OUT + "/INDEX.txt", idx);
  printf("done: %zu records in %s\n", INDEX.size(), OUT.c_str());
  return 0;
}
