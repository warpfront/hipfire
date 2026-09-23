// G12Raster2 standalone screen host: exact gfx1201 IU4 SET/ADD (symfold) raster /
// staging / epilogue variants against the byte-for-byte production control.
//   host parity <arm,...>                          tail + edge corpus, all bits vs control
//   host time <F|R> <N> <warm|cold> <arm,...> [shape,...]  full-shape parity + interleaved timing
// Device TUs are compiled separately (build.sh) with the runtime hipcc flags and
// loaded with hipModuleLoad. Card-E only (PCI c3:00.0), exact gfx1201 asserted.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x) do { auto chk_ = (x); if (chk_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(chk_)); std::exit(1); } } while (0)

struct block_i4_128 { float d; int s; unsigned char qs[64]; };
static_assert(sizeof(block_i4_128) == 72, "Xq ABI");

static const char* kHwmon = "/sys/bus/pci/devices/0000:c3:00.0/hwmon/hwmon7/";
static long read_sysfs(const char* leaf) {
    std::string p = std::string(kHwmon) + leaf;
    FILE* f = std::fopen(p.c_str(), "r");
    if (!f) return -1;
    long v = -1;
    if (std::fscanf(f, "%ld", &v) != 1) v = -1;
    std::fclose(f);
    return v;
}

// ---------- non-periodic counter hash ----------
static inline uint64_t mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static inline uint64_t h2(uint64_t a, uint64_t b) { return mix(mix(a) ^ (b * 0x632be59bd9b4e019ull)); }
static uint32_t fbits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
static float bitsf(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static uint16_t f2h(float f) {  // RNE fp32->fp16 for finite normal-range values used here
    _Float16 h = (_Float16)f; uint16_t b; std::memcpy(&b, &h, 2); return b;
}

// W: [M][K/256][136]: (sc0,zp0,sc1,zp1) fp16 then 128 nibble bytes. Symmetric zp=-8*sc.
static void gen_w(int M, int K, uint64_t seed, bool edges, std::vector<unsigned char>& w) {
    const size_t G = K / 256;
    w.resize((size_t)M * G * 136);
    for (size_t r = 0; r < (size_t)M; r++) {
        const int kind = edges ? (int)(h2(seed ^ 0x5151, r) % 13) : 12;
        for (size_t g = 0; g < G; g++) {
            unsigned char* p = w.data() + (r * G + g) * 136;
            for (int hh = 0; hh < 2; hh++) {
                const uint64_t z = h2(seed, (r * G + g) * 2 + hh);
                // |sc| in [2^-14, 2^-3), random mantissa; sign mostly positive.
                const int e = (int)(z % 11) - 14;
                const float mant = 1.0f + (float)((z >> 8) & 1023) / 1024.0f;
                float sc = std::ldexp(mant, e);
                if (((z >> 20) & 31) == 0) sc = -sc;
                if (kind == 0) sc = (g & 1) ? -0.0f : 0.0f;  // zero scales
                uint16_t sch = f2h(sc), zph = f2h(-8.0f * sc);
                std::memcpy(p + hh * 4, &sch, 2);
                std::memcpy(p + hh * 4 + 2, &zph, 2);
            }
            for (int j = 0; j < 128; j += 8) {
                uint64_t z = h2(seed + 1, ((r * G + g) * 16) + j / 8);
                std::memcpy(p + 8 + j, &z, 8);
            }
            if (kind == 1) std::memset(p + 8, 0x00, 128);          // all q=0 -> -8
            else if (kind == 2) std::memset(p + 8, 0xff, 128);     // all q=15 -> +7
            else if (kind == 3) std::memset(p + 8, 0x88, 128);     // all zero after rebias
            else if (kind == 4 && g == 0) {                        // one-hot at k in {0,127,128,255}
                std::memset(p + 8, 0x88, 128);
                const int k = (int)(r % 4 == 0 ? 0 : r % 4 == 1 ? 127 : r % 4 == 2 ? 128 : 255);
                unsigned char& b = p[8 + k / 2];
                b = (k & 1) ? (unsigned char)((b & 0x0f) | 0xf0) : (unsigned char)((b & 0xf0) | 0x0);
            }
        }
    }
}
// X: [K/128][N] block_i4_128.
static void gen_x(int N, int K, uint64_t seed, bool edges, std::vector<block_i4_128>& x) {
    const size_t H = K / 128;
    x.resize(H * (size_t)N);
    for (size_t h = 0; h < H; h++) {
        for (size_t t = 0; t < (size_t)N; t++) {
            block_i4_128& b = x[h * N + t];
            const uint64_t z = h2(seed, h * N + t);
            const int e = (int)(z % 13) - 9;
            b.d = std::ldexp(1.0f + (float)((z >> 8) & 0xfffff) / 1048576.0f, e);
            for (int j = 0; j < 64; j += 8) {
                uint64_t q = h2(seed + 7, (h * N + t) * 8 + j / 8);
                std::memcpy(b.qs + j, &q, 8);
            }
            const int kind = edges ? (int)(h2(seed ^ 0x7777, t) % 11) : 10;
            if (kind == 0) std::memset(b.qs, 0x88, 64);
            else if (kind == 1) std::memset(b.qs, 0x77, 64);
            else if (kind == 2) b.d = 0.0f;
            else if (kind == 3) std::memset(b.qs, 0x97, 64);
            b.s = 0;
            for (int j = 0; j < 64; j++) {
                const int lo = b.qs[j] & 15, hi = b.qs[j] >> 4;
                b.s += (lo < 8 ? lo : lo - 16) + (hi < 8 ? hi : hi - 16);
            }
        }
    }
}

// ---------- arms ----------
struct Arm { std::string name; hipFunction_t fset{}, fadd{}, fsilu{}; };
static Arm make_arm(const std::string& name) {
    Arm a; a.name = name;
    hipModule_t m; CHECK(hipModuleLoad(&m, ("hsaco/" + name + ".hsaco").c_str()));
    CHECK(hipModuleGetFunction(&a.fset, m, "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold"));
    CHECK(hipModuleGetFunction(&a.fadd, m, "gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold"));
    if (hipModuleGetFunction(&a.fsilu, m, "gemm_mq4g256v2_gate_up_silu_mmq_iu4_symfold") != hipSuccess) a.fsilu = nullptr;
    (void)hipGetLastError();
    for (int add = 0; add < 2; add++) {
        hipFunction_t f = add ? a.fadd : a.fset;
        int regs = 0, local = 0, occ = 0;
        CHECK(hipFuncGetAttribute(&regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, f));
        CHECK(hipFuncGetAttribute(&local, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f));
        CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, 256, 20480));
        std::printf("RESOURCE arm=%s entry=%s vgpr=%d private=%d occ_blocks_per_mp=%d\n",
                    name.c_str(), add ? "add" : "set", regs, local, occ);
        if (local != 0) { std::fprintf(stderr, "private bytes in %s\n", name.c_str()); std::exit(7); }
    }
    return a;
}

struct Dev {
    int M = 0, K = 0, N = 0;
    unsigned char* w = nullptr; unsigned char* w2 = nullptr; block_i4_128* x = nullptr;
    float* y = nullptr; float* res = nullptr;
    bool silu = false;
    void free_all() {
        for (void* p : {(void*)w, (void*)w2, (void*)x, (void*)y, (void*)res}) if (p) CHECK(hipFree(p));
        *this = Dev{};
    }
};
template <class T> static T* upload(const std::vector<T>& v) {
    T* p; CHECK(hipMalloc(&p, v.size() * sizeof(T)));
    CHECK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}
static void launch(const Arm& a, Dev& d, bool add, hipStream_t st = nullptr) {
    int M = d.M, K = d.K, N = d.N, addv = add ? 1 : 0;
    if (d.silu) {
        if (!a.fsilu) { std::fprintf(stderr, "arm %s has no SILU entry\n", a.name.c_str()); std::exit(12); }
        void* sargs[] = {&d.w, &d.w2, &d.x, &d.y, &M, &K, &N};
        CHECK(hipModuleLaunchKernel(a.fsilu, (2 * M + 127) / 128, (N + 127) / 128, 1, 256, 1, 1, 20480, st, sargs, nullptr));
        return;
    }
    void* args[] = {&d.w, &d.x, &d.y, &M, &K, &N, &addv};
    CHECK(hipModuleLaunchKernel(add ? a.fadd : a.fset, (M + 127) / 128, (N + 127) / 128, 1, 256, 1, 1,
                                20480, st, args, nullptr));
}
static void setup(Dev& d, int M, int K, int N, uint64_t seed, bool edges, bool silu = false) {
    d.M = M; d.K = K; d.N = N; d.silu = silu;
    std::vector<unsigned char> w; std::vector<block_i4_128> x;
    gen_w(M, K, seed, edges, w);
    if (silu) { std::vector<unsigned char> w2; gen_w(M, K, seed + 50, edges, w2); d.w2 = upload(w2); }
    gen_x(N, K, seed + 100, edges, x);
    d.w = upload(w); d.x = upload(x);
    const size_t n = (size_t)M * N;
    CHECK(hipMalloc(&d.y, n * 4));
    std::vector<float> res(n);
    for (size_t i = 0; i < n; i++) {
        const uint64_t z = h2(seed + 200, i);
        res[i] = std::ldexp(1.0f + (float)(z & 0xffffff) / 16777216.0f, (int)((z >> 32) % 20) - 10) * (((z >> 60) & 1) ? -1.f : 1.f);
    }
    d.res = upload(res);
}
static std::vector<float> run_once(const Arm& a, Dev& d, bool add) {
    const size_t n = (size_t)d.M * d.N;
    if (add) CHECK(hipMemcpy(d.y, d.res, n * 4, hipMemcpyDeviceToDevice));
    else CHECK(hipMemset(d.y, 0xff, n * 4));  // stale NaN: SET must overwrite every output
    launch(a, d, add);
    CHECK(hipDeviceSynchronize());
    std::vector<float> out(n);
    CHECK(hipMemcpy(out.data(), d.y, n * 4, hipMemcpyDeviceToHost));
    return out;
}
static size_t diff(const std::vector<float>& a, const std::vector<float>& b, size_t* first) {
    size_t bad = 0; *first = (size_t)-1;
    for (size_t i = 0; i < a.size(); i++)
        if (fbits(a[i]) != fbits(b[i])) { if (!bad) *first = i; bad++; }
    return bad;
}
static size_t count_nan(const std::vector<float>& a) {
    size_t c = 0; for (float v : a) if (fbits(v) == 0xffffffffu) c++; return c;
}

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out; size_t p = 0;
    while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
        if (q > p) out.push_back(s.substr(p, q - p)); p = q + 1; }
    return out;
}

static int do_parity(const std::vector<Arm>& arms) {
    const char* ref_name = std::getenv("REF") ? std::getenv("REF") : "control";
    const Arm control = make_arm(ref_name);
    std::printf("REFERENCE %s\n", ref_name);
    struct Case { int M, K, N; bool edges; };
    std::vector<Case> cases = {
        {48, 5120, 8192, true},  {1024, 5120, 4173, true}, {1000, 5120, 8155, true}, {1002, 256, 1000, true},
        {2944, 6144, 777, true}, {5120, 17408, 300, true}, {1920, 5120, 4096, true}, {256, 512, 1, true},
        {17408 - 3 * 128, 512, 2048 + 64, true}, {640, 256, 4096 + 128 * 5 + 3, true}, {384, 768, 128 * 37, false},
        {136 * 128, 256, 64 * 128, false}, {40 * 128, 256, 32 * 128 + 17, false},
    };
    int fails = 0;
    uint64_t seed = 1;
    for (const Case& c : cases) {
        Dev d; setup(d, c.M, c.K, c.N, seed++, c.edges);
        for (int add = 0; add < 2; add++) {
            const auto ref = run_once(control, d, add);
            const size_t nan_ref = add ? 0 : count_nan(ref);
            for (const Arm& a : arms) {
                const auto out = run_once(a, d, add);
                size_t first; const size_t bad = diff(ref, out, &first);
                std::printf("PARITY arm=%s %s M=%d K=%d N=%d edges=%d outputs=%zu mismatches=%zu control_stale_nan=%zu",
                            a.name.c_str(), add ? "ADD" : "SET", c.M, c.K, c.N, (int)c.edges, ref.size(), bad, nan_ref);
                if (bad) std::printf(" first=%zu ref=%a got=%a", first, ref[first], out[first]);
                std::printf("\n");
                if (bad || nan_ref) fails++;
            }
        }
        d.free_all();
    }
    if (control.fsilu) {
        struct SCase { int M, K, N; bool edges; };
        const SCase sc[] = {{1024, 5120, 4173, true}, {2048, 256, 8192, true}, {17408, 512, 643, false}, {128, 768, 1, true}};
        for (const SCase& c : sc) {
            Dev d; setup(d, c.M, c.K, c.N, seed++, c.edges, true);
            const auto ref = run_once(control, d, false);
            const size_t nan_ref = count_nan(ref);
            for (const Arm& a : arms) {
                const auto out = run_once(a, d, false);
                size_t first; const size_t bad = diff(ref, out, &first);
                std::printf("PARITY arm=%s SILU M=%d K=%d N=%d edges=%d outputs=%zu mismatches=%zu control_stale_nan=%zu",
                            a.name.c_str(), c.M, c.K, c.N, (int)c.edges, ref.size(), bad, nan_ref);
                if (bad) std::printf(" first=%zu ref=%a got=%a", first, ref[first], out[first]);
                std::printf("\n");
                if (bad || nan_ref) fails++;
            }
            d.free_all();
        }
    }
    std::printf("PARITY_SUMMARY cases=%zu arms=%zu fails=%d\n", cases.size(), arms.size(), fails);
    return fails ? 10 : 0;
}

struct Shape { const char* name; const char* group; bool add; int M, K; int calls; bool silu = false; };
static const Shape kShapes[] = {
    {"gate_up_SET_M17408_K5120", "gate_up", false, 17408, 5120, 128},
    {"qkvza_QKV_SET_M10240_K5120", "qkvza", false, 10240, 5120, 48},
    {"qkvza_Z_SET_M6144_K5120", "qkvza", false, 6144, 5120, 48},
    {"qkvza_BA_SET_M48_K5120", "qkvza", false, 48, 5120, 96},
    {"qkv_Q_SET_M12288_K5120", "qkv", false, 12288, 5120, 16},
    {"qkv_KV_SET_M1024_K5120", "qkv", false, 1024, 5120, 32},
    {"res_ADD_M5120_K6144", "res_k6144", true, 5120, 6144, 64},
    {"res_ADD_M5120_K17408", "res_k17408", true, 5120, 17408, 64},
    {"gate_up_SILU_M17408_K5120", "gate_up_silu", false, 17408, 5120, 64, true},
};

__global__ void scrub_kernel(uint4* p, size_t n, unsigned v) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = make_uint4(v, v + 1, v + 2, v + 3);
}
__global__ void touch_kernel(const uint4* p, size_t n, uint4* sink) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    if (acc.x == 0x12345678u && acc.y == 0x9abcdefu) sink[0] = acc;
}

static int do_time(const std::string& order, int N, const std::string& mode, std::vector<Arm> arms,
                   const std::vector<std::string>& shape_filter, int rounds) {
    const bool cold = mode == "cold";
    std::vector<const Shape*> shapes;
    for (const Shape& s : kShapes)
        if ((shape_filter.empty() && !s.silu) || std::find(shape_filter.begin(), shape_filter.end(), s.name) != shape_filter.end()
            || std::find(shape_filter.begin(), shape_filter.end(), s.group) != shape_filter.end()) shapes.push_back(&s);
    if (order == "R") std::reverse(shapes.begin(), shapes.end());
    // arms[0] must be control; it is the parity reference.
    const char* ref_name = std::getenv("REF") ? std::getenv("REF") : "control";
    if (arms.empty() || arms[0].name != ref_name) { std::fprintf(stderr, "first arm must be the reference %s\n", ref_name); return 2; }
    const size_t scrub_n = (512ull << 20) / 16;
    uint4* scrub = nullptr; uint4* sink = nullptr;
    if (cold) { CHECK(hipMalloc(&scrub, scrub_n * 16)); CHECK(hipMalloc(&sink, 16)); }
    for (const Shape* sp : shapes) {
        const Shape& s = *sp;
        Dev d; setup(d, s.M, s.K, N, 0xabc0 + s.M + s.K, false, s.silu);
        const double ops = 2.0 * s.M * (double)s.K * N;
        const auto ref = run_once(arms[0], d, s.add);
        for (size_t ai = 1; ai < arms.size(); ai++) {
            const auto out = run_once(arms[ai], d, s.add);
            size_t first; const size_t bad = diff(ref, out, &first);
            std::printf("FULLPARITY N=%d shape=%s arm=%s outputs=%zu mismatches=%zu\n", N, s.name, arms[ai].name.c_str(), ref.size(), bad);
            if (bad) { std::fprintf(stderr, "exact arm mismatch: abort timing\n"); return 11; }
        }
        const size_t ybytes = (size_t)s.M * N * 4;
        const size_t xbytes = (size_t)(s.K / 128) * N * 72;
        auto prep = [&](void) {
            if (s.add) CHECK(hipMemcpyAsync(d.y, d.res, ybytes, hipMemcpyDeviceToDevice, nullptr));
            if (cold) {
                hipLaunchKernelGGL(scrub_kernel, dim3(4096), dim3(256), 0, nullptr, scrub, scrub_n, 7u);
                if (s.add) CHECK(hipMemcpyAsync(d.y, d.res, ybytes, hipMemcpyDeviceToDevice, nullptr));
                hipLaunchKernelGGL(touch_kernel, dim3(4096), dim3(256), 0, nullptr, (const uint4*)d.x, xbytes / 16, sink);
            }
        };
        // Warmup >= 1.5 s, arms interleaved.
        const auto w0 = std::chrono::steady_clock::now();
        int wl = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 1.5 || wl < 4) {
            for (const Arm& a : arms) { prep(); launch(a, d, s.add); }
            if (++wl % 2 == 0) CHECK(hipDeviceSynchronize());
        }
        CHECK(hipDeviceSynchronize());
        const size_t A = arms.size();
        std::vector<std::vector<float>> ms(A);
        std::vector<long> clk, tmp, pwr;
        for (int r = 0; r < rounds; r++) {
            for (size_t k = 0; k < A; k++) {
                const size_t ai = (r & 1) ? (A - 1 - k) : k;  // alternate arm order each round
                prep();
                hipEvent_t b, e; CHECK(hipEventCreate(&b)); CHECK(hipEventCreate(&e));
                CHECK(hipEventRecord(b, nullptr));
                launch(arms[ai], d, s.add);
                CHECK(hipEventRecord(e, nullptr));
                if (ai == 0 && (r % 3) == 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(300));
                    clk.push_back(read_sysfs("freq1_input") / 1000000);
                    pwr.push_back(read_sysfs("power1_average") / 1000000);
                }
                CHECK(hipEventSynchronize(e));
                float t; CHECK(hipEventElapsedTime(&t, b, e)); ms[ai].push_back(t);
                CHECK(hipEventDestroy(b)); CHECK(hipEventDestroy(e));
            }
        }
        tmp.push_back(read_sysfs("temp2_input") / 1000);
        std::sort(clk.begin(), clk.end()); std::sort(pwr.begin(), pwr.end());
        const long clk_med = clk.empty() ? -1 : clk[clk.size() / 2];
        const long pwr_med = pwr.empty() ? -1 : pwr[pwr.size() / 2];
        for (size_t ai = 0; ai < A; ai++) {
            std::vector<float> so = ms[ai]; std::sort(so.begin(), so.end());
            std::vector<double> ratio;
            for (int r = 0; r < rounds; r++) ratio.push_back((double)ms[0][r] / ms[ai][r]);
            std::sort(ratio.begin(), ratio.end());
            const double med = so[so.size() / 2];
            std::printf("TIME order=%s N=%d mode=%s shape=%s calls=%d arm=%s median_ms=%.4f min_ms=%.4f max_ms=%.4f TOPS=%.2f speedup_vs_control_med=%.4f sclk_med_mhz=%ld power_w=%ld junction_c=%ld\n",
                        order.c_str(), N, mode.c_str(), s.name, s.calls, arms[ai].name.c_str(), med, so.front(), so.back(),
                        ops / med / 1e9, ratio[ratio.size() / 2], clk_med, pwr_med, tmp.back());
        }
        std::fflush(stdout);
        d.free_all();
    }
    if (scrub) { CHECK(hipFree(scrub)); CHECK(hipFree(sink)); }
    return 0;
}

int main(int argc, char** argv) {
    int ndev = 0; CHECK(hipGetDeviceCount(&ndev));
    if (ndev != 1) { std::fprintf(stderr, "expected exactly one visible device, got %d\n", ndev); return 3; }
    hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
    char bus[64]; CHECK(hipDeviceGetPCIBusId(bus, sizeof bus, 0));
    std::printf("DEVICE %s arch=%s mps=%d pci=%s\n", p.name, p.gcnArchName, p.multiProcessorCount, bus);
    if (std::strncmp(p.gcnArchName, "gfx1201", 7) != 0) { std::fprintf(stderr, "not gfx1201\n"); return 4; }
    for (char* c = bus; *c; c++) *c = (char)std::tolower(*c);
    if (!std::strstr(bus, "c3:00.0")) { std::fprintf(stderr, "not card-E (c3:00.0): %s\n", bus); return 5; }
    if (argc < 3) { std::fprintf(stderr, "usage\n"); return 2; }
    const std::string mode = argv[1];
    if (mode == "parity") {
        std::vector<Arm> arms; for (auto& n : split(argv[2])) arms.push_back(make_arm(n));
        return do_parity(arms);
    }
    if (mode == "time" && argc >= 6) {
        std::vector<Arm> arms; for (auto& n : split(argv[5])) arms.push_back(make_arm(n));
        std::vector<std::string> shapes; if (argc >= 7) shapes = split(argv[6]);
        const int rounds = argc >= 8 ? std::atoi(argv[7]) : 9;
        return do_time(argv[2], std::atoi(argv[3]), argv[4], arms, shapes, rounds);
    }
    std::fprintf(stderr, "bad args\n");
    return 2;
}
