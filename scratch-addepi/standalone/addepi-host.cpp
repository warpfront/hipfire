// AddEpilogue standalone: V2B ADD residual-epilogue levers on Halo/XTX.
//   oracle : bit-exactness of every GEMM variant vs the production ADD, and of
//            SET + folded norm vs ADD + production norm (Y and block_i4_128).
//   time fwd|rev [arms...] : per-arm median kernel time (flush before each).
// Objects: $BIN/v2b_<var>_<arch>.hsaco, $BIN/norm_{base,fold}_<arch>.hsaco.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "oracle_inputs.hpp"

#define CK(x)                                                                   \
    do {                                                                        \
        hipError_t e_ = (x);                                                    \
        if (e_ != hipSuccess) {                                                 \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,           \
                    hipGetErrorString(e_));                                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

struct XBlock {
    float d;
    int s;
    unsigned char qs[64];
};

static std::string g_bin, g_arch;
static std::map<std::string, hipModule_t> g_mods;

static hipFunction_t fn(const std::string& obj, const char* sym) {
    auto it = g_mods.find(obj);
    if (it == g_mods.end()) {
        hipModule_t m;
        std::string p = g_bin + "/" + obj + "_" + g_arch + ".hsaco";
        CK(hipModuleLoad(&m, p.c_str()));
        it = g_mods.emplace(obj, m).first;
    }
    hipFunction_t f;
    CK(hipModuleGetFunction(&f, it->second, sym));
    return f;
}

struct Shape {
    const char* name;
    int M, K, N;
};
static std::vector<Shape> kShapes = {{"add_k6144", 5120, 6144, 8192},
                                     {"add_k17408", 5120, 17408, 8192}};

struct Bufs {
    char *A = nullptr, *X = nullptr;
    float *Y = nullptr, *Y0 = nullptr, *T = nullptr;
};

static void launch_gemm(hipFunction_t f, const Shape& s, const Bufs& b, float* y,
                        hipStream_t st) {
    const char* A = b.A;
    const char* X = b.X;
    int M = s.M, K = s.K, N = s.N;
    void* args[] = {&A, &X, &y, &M, &K, &N};
    CK(hipModuleLaunchKernel(f, N / 256, M / 256, 1, 32, 16, 1, 65536, st, args,
                             nullptr));
}

// Norm operands (K = hidden = M of the ADD GEMM).
struct Norm {
    float *w, *awq, *s1, *s2;
    char* xi4;
    int K, N;
};

static void launch_norm(hipFunction_t f, bool fold, const Norm& n, float* x,
                        const float* delta, hipStream_t st) {
    float* xrot = nullptr;
    float eps = 1e-6f;
    int K = n.K, N = n.N;
    const float* d = delta;
    if (fold) {
        void* args[] = {&x, &d, (void*)&n.w, (void*)&n.awq, (void*)&n.s1, (void*)&n.s2,
                        &xrot, (void*)&n.xi4, &K, &eps, &N};
        CK(hipModuleLaunchKernel(f, N, 1, 1, 256, 1, 1, 1024, st, args, nullptr));
    } else {
        void* args[] = {&x, (void*)&n.w, (void*)&n.awq, (void*)&n.s1, (void*)&n.s2,
                        &xrot, (void*)&n.xi4, &K, &eps, &N};
        CK(hipModuleLaunchKernel(f, N, 1, 1, 256, 1, 1, 1024, st, args, nullptr));
    }
}

static Bufs make_bufs(const Shape& s) {
    Bufs b;
    std::vector<unsigned char> w;
    std::vector<XBlock> x;
    xreuse_oracle::fill_inputs(s.M, s.K, s.N, w, x);
    CK(hipMalloc(&b.A, w.size()));
    CK(hipMemcpy(b.A, w.data(), w.size(), hipMemcpyHostToDevice));
    CK(hipMalloc(&b.X, x.size() * 72));
    CK(hipMemcpy(b.X, x.data(), x.size() * 72, hipMemcpyHostToDevice));
    const size_t ny = (size_t)s.M * s.N;
    std::vector<float> y(ny);
    for (size_t i = 0; i < ny; ++i) y[i] = xreuse_oracle::residual(i);
    CK(hipMalloc(&b.Y, ny * 4));
    CK(hipMalloc(&b.Y0, ny * 4));
    CK(hipMalloc(&b.T, ny * 4));
    CK(hipMemcpy(b.Y0, y.data(), ny * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(b.Y, b.Y0, ny * 4, hipMemcpyDeviceToDevice));
    return b;
}

static void free_bufs(Bufs& b) {
    CK(hipFree(b.A));
    CK(hipFree(b.X));
    CK(hipFree(b.Y));
    CK(hipFree(b.Y0));
    CK(hipFree(b.T));
}

static Norm make_norm(int K, int N) {
    Norm n;
    n.K = K;
    n.N = N;
    std::vector<float> w(K), a(K), s1(256), s2(256);
    for (int i = 0; i < K; ++i) {
        const auto r = xreuse_oracle::mix64(0x1234 + i);
        w[i] = 0.5f + (float)(r & 1023) / 1024.0f;
        a[i] = 0.75f + (float)((r >> 10) & 511) / 1024.0f;
    }
    for (int i = 0; i < 256; ++i) {
        const auto r = xreuse_oracle::mix64(0x9876 + i);
        s1[i] = (r & 1) ? 1.f : -1.f;
        s2[i] = (r & 2) ? 1.f : -1.f;
    }
    CK(hipMalloc(&n.w, K * 4));
    CK(hipMalloc(&n.awq, K * 4));
    CK(hipMalloc(&n.s1, 1024));
    CK(hipMalloc(&n.s2, 1024));
    CK(hipMalloc(&n.xi4, (size_t)(K / 128) * N * 72));
    CK(hipMemcpy(n.w, w.data(), K * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(n.awq, a.data(), K * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(n.s1, s1.data(), 1024, hipMemcpyHostToDevice));
    CK(hipMemcpy(n.s2, s2.data(), 1024, hipMemcpyHostToDevice));
    return n;
}

static std::vector<char> dl(const void* p, size_t n) {
    std::vector<char> h(n);
    CK(hipMemcpy(h.data(), p, n, hipMemcpyDeviceToHost));
    return h;
}

static size_t ndiff32(const std::vector<char>& a, const std::vector<char>& b) {
    size_t d = 0;
    for (size_t i = 0; i < a.size(); i += 4) d += memcmp(&a[i], &b[i], 4) != 0;
    return d;
}

static const char* kAdd = "gemm_mq4g256v2_residual_iu4_v2b_add_gfx11";
static const char* kSet = "gemm_mq4g256v2_residual_iu4_v2b_set_gfx11";
static const char* kNorm = "fused_rmsnorm_mq_rotate_awq_i4";
// ADDEPI_PROD=1: production objects (v2b_prod = the committed V2B TU,
// norm_{base,fold}_prod = the kernels.rs concatenations).
static bool g_prod = false;
static std::string gbase() { return g_prod ? "v2b_prod" : "v2b_base"; }
static std::string gobj(const std::string& v) { return "v2b_" + v; }
static const char* gsym_add(const std::string& v) {
    return v == "prod" ? "gemm_mq4g256v2_residual_iu4_v2b_add_touch_gfx11" : kAdd;
}
static std::string nobj(bool fold) {
    return std::string(fold ? "norm_fold" : "norm_base") + (g_prod ? "_prod" : "");
}
static const char* nsym(bool fold) {
    return g_prod && fold ? "fused_rmsnorm_mq_rotate_awq_i4_fold" : kNorm;
}
// ADDEPI_REF=1: the oracle reference (ADD + norm) comes from v2b_ref / norm_ref,
// objects built from the pre-lever base sources, so the lever objects (e.g. the
// runtime's JIT cache) are compared against the incumbent, not against their
// own `_add` entry.
static bool g_ref = false;
static std::string rgemm() { return g_ref ? "v2b_ref" : gbase(); }
static std::string rnorm() { return g_ref ? "norm_ref" : nobj(false); }

static std::vector<std::string> g_vars;  // GEMM ADD variants beyond base

static int oracle() {
    size_t total = 0, bad = 0;
    for (const Shape& s : kShapes) {
        Bufs b = make_bufs(s);
        const size_t ny = (size_t)s.M * s.N;
        Norm n = make_norm(s.M, s.N);
        const size_t ni4 = (size_t)(s.M / 128) * s.N * 72;
        // Reference: production ADD, then production norm on the sum.
        CK(hipMemcpy(b.Y, b.Y0, ny * 4, hipMemcpyDeviceToDevice));
        launch_gemm(fn(rgemm(), kAdd), s, b, b.Y, 0);
        CK(hipDeviceSynchronize());
        auto yref = dl(b.Y, ny * 4);
        launch_norm(fn(rnorm(), kNorm), false, n, b.Y, nullptr, 0);
        CK(hipDeviceSynchronize());
        auto i4ref = dl(n.xi4, ni4);
        if (g_ref) {  // lever build's unfolded norm (pre-attention path) vs reference
            CK(hipMemset(n.xi4, 0x5a, ni4));
            launch_norm(fn(nobj(false), kNorm), false, n, b.Y, nullptr, 0);
            CK(hipDeviceSynchronize());
            auto i4b = dl(n.xi4, ni4);
            size_t di = 0;
            for (size_t i = 0; i < ni4; ++i) di += i4b[i] != i4ref[i];
            printf("oracle %-10s NORM base   block_i4_128 bytes %zu differing %zu\n", s.name, ni4, di);
            total += ni4;
            bad += di;
        }
        for (const auto& v : g_vars) {
            CK(hipMemcpy(b.Y, b.Y0, ny * 4, hipMemcpyDeviceToDevice));
            launch_gemm(fn(gobj(v), gsym_add(v)), s, b, b.Y, 0);
            CK(hipDeviceSynchronize());
            const size_t d = ndiff32(dl(b.Y, ny * 4), yref);
            printf("oracle %-10s ADD %-8s outputs %zu differing %zu\n", s.name, v.c_str(), ny, d);
            total += ny;
            bad += d;
        }
        // Fold: SET into T, then the folded norm updates Y in place.
        CK(hipMemcpy(b.Y, b.Y0, ny * 4, hipMemcpyDeviceToDevice));
        CK(hipMemset(n.xi4, 0x5a, ni4));
        launch_gemm(fn(gbase(), kSet), s, b, b.T, 0);
        launch_norm(fn(nobj(true), nsym(true)), true, n, b.Y, b.T, 0);
        CK(hipDeviceSynchronize());
        const size_t dy = ndiff32(dl(b.Y, ny * 4), yref);
        auto i4 = dl(n.xi4, ni4);
        size_t di = 0;
        for (size_t i = 0; i < ni4; ++i) di += i4[i] != i4ref[i];
        printf("oracle %-10s FOLD residual outputs %zu differing %zu; block_i4_128 bytes %zu differing %zu\n",
               s.name, ny, dy, ni4, di);
        total += ny + ni4;
        bad += dy + di;
        free_bufs(b);
    }
    printf("oracle total compared %zu differing %zu -> %s\n", total, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

static double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static int timing(bool rev, int reps) {
    void* flush;
    const size_t fl = 256u << 20;
    CK(hipMalloc(&flush, fl));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    for (const Shape& s : kShapes) {
        Bufs b = make_bufs(s);
        Norm n = make_norm(s.M, s.N);
        struct Arm {
            std::string name;
            std::function<void()> go;
        };
        std::vector<Arm> arms;
        arms.push_back({"add:base", [&] { launch_gemm(fn(gbase(), kAdd), s, b, b.Y, 0); }});
        arms.push_back({"set:base", [&] { launch_gemm(fn(gbase(), kSet), s, b, b.T, 0); }});
        for (const auto& v : g_vars)
            arms.push_back({"add:" + v, [&, v] { launch_gemm(fn(gobj(v), gsym_add(v)), s, b, b.Y, 0); }});
        arms.push_back({"norm:base", [&] { launch_norm(fn(nobj(false), nsym(false)), false, n, b.Y, nullptr, 0); }});
        arms.push_back({"norm:fold", [&] { launch_norm(fn(nobj(true), nsym(true)), true, n, b.Y, b.T, 0); }});
        if (rev) std::reverse(arms.begin(), arms.end());
        std::map<std::string, double> res;
        for (auto& a : arms) {
            // Warm-up >= 0.5 s.
            auto t0 = std::chrono::steady_clock::now();
            do {
                a.go();
                CK(hipDeviceSynchronize());
            } while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500));
            std::vector<float> ms;
            for (int r = 0; r < reps; ++r) {
                CK(hipMemsetAsync(flush, r & 0xff, fl, 0));
                CK(hipEventRecord(e0, 0));
                a.go();
                CK(hipEventRecord(e1, 0));
                CK(hipEventSynchronize(e1));
                float t;
                CK(hipEventElapsedTime(&t, e0, e1));
                ms.push_back(t);
            }
            res[a.name] = median(ms);
            printf("time %s %-10s %-12s %8.3f ms\n", rev ? "rev" : "fwd", s.name, a.name.c_str(),
                   res[a.name]);
            fflush(stdout);
        }
        const double base = res["add:base"] + res["norm:base"];
        const double fold = res["set:base"] + res["norm:fold"];
        printf("summary %s %s add+norm %.3f  set+foldnorm %.3f  fold saves %.3f ms (%.2f%%)\n",
               rev ? "rev" : "fwd", s.name, base, fold, base - fold, 100.0 * (base - fold) / base);
        free_bufs(b);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: addepi-host oracle|time fwd|rev\n");
        return 2;
    }
    const char* b = getenv("ADDEPI_BIN");
    g_bin = b ? b : "bin";
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, 0));
    g_arch = std::string(p.gcnArchName).substr(0, std::string(p.gcnArchName).find(':'));
    const char* want = getenv("EXPECTED_ARCH");
    printf("device %s, %d MPs\n", g_arch.c_str(), p.multiProcessorCount);
    if (want && g_arch != want) {
        fprintf(stderr, "arch %s != expected %s\n", g_arch.c_str(), want);
        return 3;
    }
    const char* vs = getenv("ADDEPI_VARS");  // e.g. "pf4,pf8"
    if (vs) {
        std::string s = vs;
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            if (j > i) g_vars.push_back(s.substr(i, j - i));
            i = j + 1;
        }
    }
    g_prod = getenv("ADDEPI_PROD") != nullptr;
    g_ref = getenv("ADDEPI_REF") != nullptr;
    if (const char* ns = getenv("ADDEPI_N")) {  // token count override (e.g. 512)
        for (auto& sh : kShapes) sh.N = atoi(ns);
    }
    if (!strcmp(argv[1], "oracle")) return oracle();
    if (!strcmp(argv[1], "time")) return timing(argc > 2 && !strcmp(argv[2], "rev"), 7);
    return 2;
}
