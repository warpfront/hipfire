// G12Port scratch host (host-only TU; device code from scratch-g12port/mods,
// built by build_mods.sh with the JIT's concatenation and hipcc argv).
// Modes:
//   oracle                 exactness of every ported lever vs its base arm
//   time <fwd|rev> [N]     arm timings (median of 7 HIP-event reps)
// Levers / arms:
//   F1   A: base SET(gate)+SET(up) + base silu producer(g,u)
//        C: gate_up_silu(G,U -> h) + h-producer(h)
//   ZBA  A: base SET(Z) + SET(beta) + SET(alpha)
//        C: zba SET([Z|beta|alpha|pad]) scattering beta/alpha
//   KEEP A: fused_rmsnorm_mq_rotate_awq_i4_gfx12   C: ..._keep_x_gfx12
//   GRID A: gemv_mq4g256v2_residual grid M         C: grid ceil(M/2)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "../../wt-stack-wave/scratch-f1lite/oracle_inputs.hpp"

struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};
static_assert(sizeof(block_i4_128) == 72, "ABI");

#define OK(x)                                                                 \
    do {                                                                      \
        hipError_t s_ = (x);                                                  \
        if (s_ != hipSuccess) {                                               \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,         \
                    hipGetErrorString(s_));                                   \
            exit(2);                                                          \
        }                                                                     \
    } while (0)

static std::string g_mods = "scratch-g12port/mods";
static hipFunction_t fn(const char* mod, const char* name) {
    static std::vector<std::pair<std::string, hipModule_t>> cache;
    hipModule_t m = nullptr;
    for (auto& e : cache)
        if (e.first == mod) m = e.second;
    if (!m) {
        OK(hipModuleLoad(&m, (g_mods + "/" + mod + ".co").c_str()));
        cache.push_back({mod, m});
    }
    hipFunction_t f;
    OK(hipModuleGetFunction(&f, m, name));
    return f;
}

static hipFunction_t F_SET_BASE, F_SET_NEW, F_SILU_GEMM, F_ZBA, F_PROD, F_HIN, F_RMS, F_KEEP, F_GEMV;
static void init() {
    F_SET_BASE = fn("gemm_base", "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold");
    F_SET_NEW = fn("gemm_new", "gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold");
    F_SILU_GEMM = fn(getenv("SILU_MOD") ? getenv("SILU_MOD") : "gemm_new", "gemm_mq4g256v2_gate_up_silu_mmq_iu4_symfold");
    F_ZBA = fn("gemm_new", "gemm_mq4g256v2_zba_set_mmq_iu4_symfold");
    F_PROD = fn("silu_base", "fused_silu_mul_mq_rotate_awq_i4_gfx12");
    F_HIN = fn("silu_hin", "fused_silu_mul_mq_rotate_awq_i4_hin_gfx12");
    F_RMS = fn("rms_base", "fused_rmsnorm_mq_rotate_awq_i4_gfx12");
    F_KEEP = fn("rms_keep", "fused_rmsnorm_mq_rotate_awq_i4_keep_x_gfx12");
    F_GEMV = fn("gemv_res", "gemv_mq4g256v2_residual");
}

template <class T> static T* dalloc(size_t n) {
    T* p;
    OK(hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}
template <class T> static T* upload(const std::vector<T>& v) {
    T* p = dalloc<T>(v.size());
    OK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}
template <class T> static std::vector<T> download(const T* p, size_t n) {
    std::vector<T> v(n);
    OK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}
static uint64_t mix(uint64_t x) { return xreuse_oracle::mix64(x); }
static std::vector<float> signs(uint64_t seed, int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = (mix(seed + i) & 1) ? 1.0f : -1.0f;
    return v;
}
static std::vector<float> uniform(uint64_t seed, size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = lo + (hi - lo) * (float)((mix(seed + i) >> 11) * (1.0 / 9007199254740992.0));
    return v;
}
// Weights for M rows at seed offset `row0` (distinct matrices stay distinct).
static std::vector<unsigned char> weights(int M, int K, uint64_t salt) {
    std::vector<unsigned char> w;
    std::vector<block_i4_128> x;
    xreuse_oracle::fill_inputs(M, K, 1, w, x);
    // Re-salt the payload so gate/up/Z/beta/alpha differ.
    for (size_t i = 8; i < w.size(); ++i)
        if ((i % 136) >= 8) w[i] ^= (unsigned char)mix(salt * 0x9e3779b97f4a7c15ULL + i);
    return w;
}
static std::vector<block_i4_128> acts(int K, int N, float dscale) {
    std::vector<unsigned char> w;
    std::vector<block_i4_128> x;
    xreuse_oracle::fill_inputs(1, K, N, w, x);
    for (auto& b : x) b.d *= dscale;
    return x;
}

// ------------------------------------------------------------- launches
struct Args {
    std::vector<unsigned char> buf;
    template <class T> Args& p(T v) {
        size_t a = alignof(T), o = (buf.size() + a - 1) / a * a;
        buf.resize(o + sizeof(T));
        memcpy(buf.data() + o, &v, sizeof(T));
        return *this;
    }
};
static void launch(hipFunction_t f, dim3 g, dim3 b, unsigned lds, Args& a) {
    size_t sz = a.buf.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.buf.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE,
                   &sz, HIP_LAUNCH_PARAM_END};
    OK(hipModuleLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, lds, nullptr, nullptr, cfg));
}
static void set(hipFunction_t f, const void* A, const void* X, float* Y, int M, int K, int N) {
    Args a;
    a.p(A).p(X).p(Y).p(M).p(K).p(N).p(0);
    launch(f, dim3((M + 127) / 128, (N + 127) / 128), dim3(256), 20480, a);
}
static void silu_gemm(const void* G, const void* U, const void* X, float* H, int M, int K, int N) {
    Args a;
    a.p(G).p(U).p(X).p(H).p(M).p(K).p(N);
    launch(F_SILU_GEMM, dim3(2 * M / 128, (N + 127) / 128), dim3(256), 20480, a);
}
static void zba(const void* A, const void* X, float* Z, float* B, float* Al, int M, int K, int N,
                int zm, int bam) {
    Args a;
    a.p(A).p(X).p(Z).p(B).p(Al).p(M).p(K).p(N).p(zm).p(bam);
    launch(F_ZBA, dim3((M + 127) / 128, (N + 127) / 128), dim3(256), 20480, a);
}
struct Sig {
    float *s1, *s2;
};
static void prod(const float* g, const float* u, const float* awq, Sig s, float* xr, void* i4,
                 int K, int N) {
    Args a;
    a.p(g).p(u).p(awq).p(s.s1).p(s.s2).p(xr).p(i4).p(K).p(N);
    launch(F_PROD, dim3(K / 256, N), dim3(32), 0, a);
}
static void hin(const float* h, const float* awq, Sig s, float* xr, void* i4, int K, int N) {
    Args a;
    a.p(h).p(awq).p(s.s1).p(s.s2).p(xr).p(i4).p(K).p(N);
    launch(F_HIN, dim3(K / 256, N), dim3(32), 0, a);
}
static void rms(hipFunction_t f, bool keep, const float* x, const float* w, const float* awq, Sig s,
                float* xr, void* i4, int K, int N) {
    Args a;
    a.p(x).p(w).p(awq).p(s.s1).p(s.s2).p(xr).p(i4).p(K).p(1e-6f).p(N);
    unsigned lds = (xr || keep) ? (unsigned)((K + 256) * 4) : 1024u;
    launch(f, dim3(N), dim3(256), lds, a);
}
static void gemv(const void* A, const float* x, float* y, int M, int K, bool tight) {
    Args a;
    a.p(A).p(x).p(y).p(M).p(K);
    launch(F_GEMV, dim3(tight ? (M + 1) / 2 : M), dim3(32), 0, a);
}

// ------------------------------------------------------------- oracle
static bool g_fail = false;
template <class T> static void cmp(const char* what, const T* a, const T* b, size_t n) {
    auto va = download(a, n), vb = download(b, n);
    size_t diff = 0, first = n;
    for (size_t i = 0; i < n; ++i)
        if (memcmp(&va[i], &vb[i], sizeof(T))) {
            if (!diff) first = i;
            ++diff;
        }
    printf("  %-44s n=%zu differing=%zu%s\n", what, n, diff,
           diff ? (" first=" + std::to_string(first)).c_str() : "");
    if (diff) g_fail = true;
}
static size_t i4_bytes(int K, int N) { return (size_t)(K / 128) * N * sizeof(block_i4_128); }

static void f1_case(int M, int K, int N, float dscale, const char* tag) {
    printf("F1 %s M=%d K=%d N=%d dscale=%g\n", tag, M, K, N, dscale);
    auto wg = weights(M, K, 11), wu = weights(M, K, 12);
    auto x = acts(K, N, dscale);
    auto G = upload(wg), U = upload(wu);
    auto X = upload(x);
    auto awq = upload(uniform(7, M, 0.5f, 2.0f));
    Sig s{upload(signs(42, 256)), upload(signs(1042, 256))};
    float *g = dalloc<float>((size_t)N * M), *u = dalloc<float>((size_t)N * M),
          *h = dalloc<float>((size_t)N * M);
    float *xa = dalloc<float>((size_t)N * M), *xc = dalloc<float>((size_t)N * M);
    auto *ia = dalloc<unsigned char>(i4_bytes(M, N)), *ic = dalloc<unsigned char>(i4_bytes(M, N));
    OK(hipMemset(g, 0xCD, (size_t)N * M * 4));
    OK(hipMemset(u, 0xCD, (size_t)N * M * 4));
    OK(hipMemset(h, 0xCD, (size_t)N * M * 4));
    set(F_SET_BASE, G, X, g, M, K, N);
    set(F_SET_BASE, U, X, u, M, K, N);
    silu_gemm(G, U, X, h, M, K, N);
    // Count exp-overflow silu specials in the reference gate stream.
    auto gv = download(g, (size_t)N * M);
    size_t big = 0;
    for (float v : gv) big += std::fabs(v) > 88.0f;
    printf("  |g|>88 count %zu\n", big);
    for (int emit = 0; emit < 2; ++emit) {
        OK(hipMemset(ia, 0x5A, i4_bytes(M, N)));
        OK(hipMemset(ic, 0x5A, i4_bytes(M, N)));
        OK(hipMemset(xa, 0x77, (size_t)N * M * 4));
        OK(hipMemset(xc, 0x77, (size_t)N * M * 4));
        prod(g, u, awq, s, emit ? xa : nullptr, ia, M, N);
        hin(h, awq, s, emit ? xc : nullptr, ic, M, N);
        OK(hipDeviceSynchronize());
        cmp(emit ? "block_i4_128 (x_rot emitted)" : "block_i4_128 (no x_rot)", ia, ic, i4_bytes(M, N));
        if (emit) cmp("x_rot f32 (A producer vs h-producer)", xa, xc, (size_t)N * M);
    }
    OK(hipFree(G)); OK(hipFree(U)); OK(hipFree(X)); OK(hipFree(awq));
    OK(hipFree(s.s1)); OK(hipFree(s.s2)); OK(hipFree(g)); OK(hipFree(u)); OK(hipFree(h));
    OK(hipFree(xa)); OK(hipFree(xc)); OK(hipFree(ia)); OK(hipFree(ic));
}

static void zba_case(int zm, int bam, int K, int N, const char* tag) {
    const int Mf = (zm + 2 * bam + 127) / 128 * 128;
    printf("ZBA %s zm=%d ba=%d Mfold=%d K=%d N=%d\n", tag, zm, bam, Mf, K, N);
    auto wz = weights(zm, K, 21), wb = weights(bam, K, 22), wa = weights(bam, K, 23);
    const size_t rb = (size_t)(K / 256) * 136;
    std::vector<unsigned char> wf(Mf * rb, 0);
    memcpy(wf.data(), wz.data(), wz.size());
    memcpy(wf.data() + zm * rb, wb.data(), wb.size());
    memcpy(wf.data() + (zm + bam) * rb, wa.data(), wa.size());
    auto Z = upload(wz), B = upload(wb), Al = upload(wa), F = upload(wf);
    auto X = upload(acts(K, N, 1.0f));
    float *za = dalloc<float>((size_t)N * zm), *zc = dalloc<float>((size_t)N * zm);
    float *ba = dalloc<float>((size_t)N * bam), *bc = dalloc<float>((size_t)N * bam);
    float *aa = dalloc<float>((size_t)N * bam), *ac = dalloc<float>((size_t)N * bam);
    for (auto p : {za, zc}) OK(hipMemset(p, 0xCD, (size_t)N * zm * 4));
    for (auto p : {ba, bc, aa, ac}) OK(hipMemset(p, 0xCD, (size_t)N * bam * 4));
    set(F_SET_BASE, Z, X, za, zm, K, N);
    set(F_SET_BASE, B, X, ba, bam, K, N);
    set(F_SET_BASE, Al, X, aa, bam, K, N);
    zba(F, X, zc, bc, ac, Mf, K, N, zm, bam);
    OK(hipDeviceSynchronize());
    cmp("Z", za, zc, (size_t)N * zm);
    cmp("beta", ba, bc, (size_t)N * bam);
    cmp("alpha", aa, ac, (size_t)N * bam);
    // New-module SET must equal base SET (instruction-identical entry).
    set(F_SET_NEW, Z, X, zc, zm, K, N);
    OK(hipDeviceSynchronize());
    cmp("SET new module vs base module", za, zc, (size_t)N * zm);
    for (auto p : {(void*)Z, (void*)B, (void*)Al, (void*)F, (void*)X, (void*)za, (void*)zc,
                   (void*)ba, (void*)bc, (void*)aa, (void*)ac})
        OK(hipFree(p));
}

static void keep_case(int K, int N, const char* tag) {
    printf("KEEP %s K=%d N=%d\n", tag, K, N);
    auto x = upload(uniform(31, (size_t)N * K, -3.0f, 3.0f));
    auto w = upload(uniform(32, K, 0.2f, 1.8f));
    auto awq = upload(uniform(33, K, 0.5f, 2.0f));
    Sig s{upload(signs(42, 256)), upload(signs(1042, 256))};
    auto *ia = dalloc<unsigned char>(i4_bytes(K, N)), *ic = dalloc<unsigned char>(i4_bytes(K, N));
    float *xa = dalloc<float>((size_t)N * K), *xc = dalloc<float>((size_t)N * K);
    for (int emit = 0; emit < 2; ++emit) {
        OK(hipMemset(ia, 0x5A, i4_bytes(K, N)));
        OK(hipMemset(ic, 0x5A, i4_bytes(K, N)));
        OK(hipMemset(xa, 0x77, (size_t)N * K * 4));
        OK(hipMemset(xc, 0x77, (size_t)N * K * 4));
        rms(F_RMS, false, x, w, awq, s, emit ? xa : nullptr, ia, K, N);
        rms(F_KEEP, true, x, w, awq, s, emit ? xc : nullptr, ic, K, N);
        OK(hipDeviceSynchronize());
        cmp(emit ? "block_i4_128 (x_rot emitted)" : "block_i4_128 (no x_rot)", ia, ic, i4_bytes(K, N));
        if (emit) cmp("x_rot f32", xa, xc, (size_t)N * K);
    }
    for (auto p : {(void*)x, (void*)w, (void*)awq, (void*)s.s1, (void*)s.s2, (void*)ia, (void*)ic,
                   (void*)xa, (void*)xc})
        OK(hipFree(p));
}

static void grid_case(int M, int K, const char* tag) {
    printf("GRID %s M=%d K=%d\n", tag, M, K);
    auto A = upload(weights(M, K, 41));
    auto x = upload(uniform(42, K, -1.0f, 1.0f));
    auto y0 = uniform(43, M, -4.0f, 4.0f);
    float *ya = upload(y0), *yc = upload(y0);
    gemv(A, x, ya, M, K, false);
    gemv(A, x, yc, M, K, true);
    OK(hipDeviceSynchronize());
    cmp("y (residual +=)", ya, yc, M);
    for (auto p : {(void*)A, (void*)x, (void*)ya, (void*)yc}) OK(hipFree(p));
}

static void oracle() {
    f1_case(17408, 5120, 1024, 1.0f, "prod-shape");
    f1_case(1024, 1024, 512, 40.0f, "wide-range");
    f1_case(1024, 1024, 512, 20000.0f, "silu-specials");
    f1_case(512, 512, 1000, 1.0f, "partial-N");
    f1_case(768, 768, 64, 8000.0f, "odd-tiles-N64");
    zba_case(6144, 48, 5120, 1024, "prod-shape");
    zba_case(6144, 48, 5120, 1000, "partial-N");
    zba_case(256, 48, 512, 64, "small");
    keep_case(5120, 1024, "prod-K");
    keep_case(5120, 77, "odd-N");
    keep_case(1024, 256, "small-K");
    grid_case(5120, 17408, "down");
    grid_case(5120, 6144, "out");
    grid_case(2049, 512, "odd-M");
    printf(g_fail ? "ORACLE_FAIL\n" : "ORACLE_PASS\n");
}

// ------------------------------------------------------------- timing
template <class F> static float time_ms(F&& body) {
    hipEvent_t a, b;
    OK(hipEventCreate(&a));
    OK(hipEventCreate(&b));
    std::vector<float> t;
    for (int i = 0; i < 3; ++i) body();  // warm
    for (int r = 0; r < 7; ++r) {
        OK(hipEventRecord(a));
        body();
        OK(hipEventRecord(b));
        OK(hipEventSynchronize(b));
        float ms;
        OK(hipEventElapsedTime(&ms, a, b));
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[3];
}
static void timing(bool rev, int N) {
    const int K = 5120, M = 17408;
    // F1 buffers (shared between arms).
    auto G = upload(weights(M, K, 11)), U = upload(weights(M, K, 12));
    auto X = upload(acts(K, N, 1.0f));
    auto awqf = upload(uniform(7, M, 0.5f, 2.0f));
    Sig s{upload(signs(42, 256)), upload(signs(1042, 256))};
    float *g = dalloc<float>((size_t)N * M), *u = dalloc<float>((size_t)N * M);
    auto* i4 = dalloc<unsigned char>(i4_bytes(M, N));
    // ZBA buffers.
    const int zm = 6144, bam = 48, Mf = 6272;
    auto Z = upload(weights(zm, K, 21)), B = upload(weights(bam, K, 22)),
         Al = upload(weights(bam, K, 23)), Fz = upload(weights(Mf, K, 24));
    float *zo = dalloc<float>((size_t)N * zm), *bo = dalloc<float>((size_t)N * bam),
          *ao = dalloc<float>((size_t)N * bam);
    // RMS buffers.
    auto xr = upload(uniform(31, (size_t)N * K, -3.0f, 3.0f));
    auto wr = upload(uniform(32, K, 0.2f, 1.8f)), awqr = upload(uniform(33, K, 0.5f, 2.0f));
    auto* i4r = dalloc<unsigned char>(i4_bytes(K, N));
    // GEMV buffers (decode down/out).
    auto Ad = upload(weights(5120, 17408, 41)), Ao = upload(weights(5120, 6144, 44));
    auto xd = upload(uniform(42, 17408, -1.0f, 1.0f));
    auto yd = upload(uniform(43, 5120, -4.0f, 4.0f));

    struct Arm { const char* name; std::function<void()> f; };
    std::vector<Arm> arms = {
        {"F1.A set+set+producer", [&] {
             set(F_SET_BASE, G, X, g, M, K, N); set(F_SET_BASE, U, X, u, M, K, N);
             prod(g, u, awqf, s, nullptr, i4, M, N); }},
        {"F1.C silu-gemm+h-producer", [&] {
             silu_gemm(G, U, X, g, M, K, N); hin(g, awqf, s, nullptr, i4, M, N); }},
        {"F1.A gemm-pair only", [&] {
             set(F_SET_BASE, G, X, g, M, K, N); set(F_SET_BASE, U, X, u, M, K, N); }},
        {"F1.C silu-gemm only", [&] { silu_gemm(G, U, X, g, M, K, N); }},
        {"F1.A producer only", [&] { prod(g, u, awqf, s, nullptr, i4, M, N); }},
        {"F1.C h-producer only", [&] { hin(g, awqf, s, nullptr, i4, M, N); }},
        {"ZBA.A Z+beta+alpha SETs", [&] {
             set(F_SET_BASE, Z, X, zo, zm, K, N); set(F_SET_BASE, B, X, bo, bam, K, N);
             set(F_SET_BASE, Al, X, ao, bam, K, N); }},
        {"ZBA.C folded SET", [&] { zba(Fz, X, zo, bo, ao, Mf, K, N, zm, bam); }},
        {"KEEP.A rms awq i4", [&] { rms(F_RMS, false, xr, wr, awqr, s, nullptr, i4r, K, N); }},
        {"KEEP.C rms awq i4 keep-x", [&] { rms(F_KEEP, true, xr, wr, awqr, s, nullptr, i4r, K, N); }},
        {"GRID.A gemv down+out grid M", [&] {
             gemv(Ad, xd, yd, 5120, 17408, false); gemv(Ao, xd, yd, 5120, 6144, false); }},
        {"GRID.C gemv down+out grid M/2", [&] {
             gemv(Ad, xd, yd, 5120, 17408, true); gemv(Ao, xd, yd, 5120, 6144, true); }},
    };
    std::vector<float> res(arms.size());
    for (size_t k = 0; k < arms.size(); ++k) {
        size_t i = rev ? arms.size() - 1 - k : k;
        res[i] = time_ms(arms[i].f);
    }
    for (size_t i = 0; i < arms.size(); ++i)
        printf("TIME N=%d %-34s %9.4f ms\n", N, arms[i].name, res[i]);
}

int main(int argc, char** argv) {
    if (const char* m = getenv("MODS")) g_mods = m;
    hipDeviceProp_t p;
    OK(hipGetDeviceProperties(&p, 0));
    printf("device %s %s\n", p.name, p.gcnArchName);
    if (strncmp(p.gcnArchName, "gfx1201", 7)) {
        fprintf(stderr, "expected gfx1201\n");
        return 2;
    }
    init();
    std::string mode = argc > 1 ? argv[1] : "oracle";
    if (mode == "oracle") {
        oracle();
        return g_fail;
    }
    if (mode == "time") {
        timing(argc > 2 && !strcmp(argv[2], "rev"), argc > 3 ? atoi(argv[3]) : 8192);
        return 0;
    }
    return 2;
}
