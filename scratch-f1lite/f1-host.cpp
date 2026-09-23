// F1Lite scratch host (host-only TU; device code in separately built HSACOs
// from the branch's production sources, see build.sh). Not production.
// Modes:
//   oracle               F1-lite exactness vs the production pair + producer
//   time <fwd|rev> [N]   arm timing, median of 7 HIP-event reps per arm
// Arms (gate/up shape M 17408 each, K 5120, N):
//   A pair : base SET(gate) + base SET(up) + base producer(g,u)   [today]
//   C silu : gate_up_silu(G, U -> h) + h-producer(h)             [F1-lite]
// Gate and up are separate allocations (as in production); the F1 kernel
// interleaves them by address only.
// Env: EXPECTED_ARCH, ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, GEMM_STEM
//      (v2c|v2b), GEMM_BASE_OBJECT, GEMM_F1_OBJECT, PROD_SILU_OBJECT,
//      PROD_HIN_OBJECT, optional PROD_SILU_JIT_OBJECT (hipx JIT object).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "oracle_inputs.hpp"

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

static double wall() {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// ---------------------------------------------------------------- modules
struct Geo {
    std::string stem;
    int tm, tn, waves, lds;
};
static Geo g_geo;
static hipFunction_t g_set_base, g_set_new, g_silu, g_prod, g_hin, g_prod_jit = nullptr;

static hipModule_t load(const char* env) {
    const char* p = getenv(env);
    if (!p) {
        fprintf(stderr, "missing %s\n", env);
        exit(3);
    }
    hipModule_t m;
    OK(hipModuleLoad(&m, p));
    printf("OBJECT %s=%s\n", env, p);
    return m;
}

static void init_modules() {
    const char* st = getenv("GEMM_STEM");
    const std::string s = st ? st : "v2c";
    if (s == "v2c") g_geo = {"v2c", 128, 128, 8, 32768};
    else if (s == "v2b") g_geo = {"v2b", 256, 256, 16, 65536};
    else exit(3);
    hipModule_t gb = load("GEMM_BASE_OBJECT"), gf = load("GEMM_F1_OBJECT");
    const std::string set = "gemm_mq4g256v2_residual_iu4_" + s + "_set_gfx11";
    OK(hipModuleGetFunction(&g_set_base, gb, set.c_str()));
    OK(hipModuleGetFunction(&g_set_new, gf, set.c_str()));
    OK(hipModuleGetFunction(&g_silu, gf, ("gemm_mq4g256v2_gate_up_silu_iu4_" + s + "_gfx11").c_str()));
    OK(hipModuleGetFunction(&g_prod, load("PROD_SILU_OBJECT"), "fused_silu_mul_mq_rotate_awq_i4"));
    OK(hipModuleGetFunction(&g_hin, load("PROD_HIN_OBJECT"), "fused_silu_mul_mq_rotate_awq_i4_hin"));
    if (getenv("PROD_SILU_JIT_OBJECT"))
        OK(hipModuleGetFunction(&g_prod_jit, load("PROD_SILU_JIT_OBJECT"),
                                "fused_silu_mul_mq_rotate_awq_i4"));
}

static void resources() {
    struct E {
        const char* n;
        hipFunction_t f;
        int threads, lds;
    } es[] = {{"gemm_set_base", g_set_base, 32 * g_geo.waves, g_geo.lds},
              {"gemm_set_new", g_set_new, 32 * g_geo.waves, g_geo.lds},
              {"gemm_gate_up_silu", g_silu, 32 * g_geo.waves, g_geo.lds},
              {"silu_producer", g_prod, 32, 0},
              {"hin_producer", g_hin, 32, 0}};
    for (auto& e : es) {
        int regs = 0, priv = 0, occ = 0;
        OK(hipFuncGetAttribute(&regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, e.f));
        OK(hipFuncGetAttribute(&priv, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, e.f));
        OK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&occ, e.f, e.threads, e.lds));
        printf("RESOURCE %s vgpr=%d private=%d dynamic_lds=%d blocks_per_mp=%d\n", e.n, regs,
               priv, e.lds, occ);
        if (priv || occ < 1) exit(4);
    }
    fflush(stdout);
}

// ---------------------------------------------------------------- data
static std::vector<float> gen_fwht_signs(uint32_t seed, int n) {
    std::vector<float> v(n);
    uint32_t state = seed;
    for (int i = 0; i < n; ++i) {
        state = (state * 1103515245u + 12345u) & 0x7fffffffu;
        v[i] = ((state >> 16) & 1) ? 1.0f : -1.0f;
    }
    return v;
}

template <class T>
static T* upload(const T* v, size_t n) {
    T* p;
    OK(hipMalloc(&p, std::max<size_t>(n * sizeof(T), 256)));
    OK(hipMemcpy(p, v, n * sizeof(T), hipMemcpyHostToDevice));
    return p;
}
template <class T>
static T* upload(const std::vector<T>& v) {
    return upload(v.data(), v.size());
}
template <class T>
static std::vector<T> download(const T* p, size_t n) {
    std::vector<T> v(n);
    OK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

// d specials widen the silu input range (exp overflow/underflow, zero, sign).
enum class XMode { Hashed, SiluSpecials };

struct Case {
    int Me, K, N;  // Me = rows per projection (gate = up = Me)
    size_t row_bytes;
    std::vector<unsigned char> w;  // [gate; up] rows
    std::vector<block_i4_128> x;
    std::vector<float> awq, s1, s2;
    char *dg = nullptr, *du = nullptr;
    block_i4_128* dx = nullptr;
    float *dawq = nullptr, *ds1 = nullptr, *ds2 = nullptr;
    float *yg = nullptr, *yu = nullptr, *yh = nullptr, *xrot = nullptr;
    block_i4_128* xi4 = nullptr;
};

static void build_case(Case& c, int Me, int K, int N, XMode xm) {
    c.Me = Me, c.K = K, c.N = N;
    xreuse_oracle::fill_inputs(2 * Me, K, N, c.w, c.x);
    c.row_bytes = (size_t)(K / 256) * 136;
    if (xm == XMode::SiluSpecials) {
        const float ds[] = {0.0f, 1e-30f, -3.5f, 3.5f, 30.0f, -30.0f, 400.0f};
        for (size_t b = 0; b < c.x.size(); ++b)
            if (b % 3 == 0) c.x[b].d = ds[(b / 3) % 7];
    }
    c.awq.resize(Me);  // producer K = Me
    for (int i = 0; i < Me; ++i) {
        const uint64_t r = xreuse_oracle::mix64(UINT64_C(0x3243f6a8885a308d) + i);
        c.awq[i] = std::ldexp(1.0f + (float)(r & 0xffff) / 65536.0f, (int)((r >> 16) % 4) - 2);
    }
    c.s1 = gen_fwht_signs(42, 256);
    c.s2 = gen_fwht_signs(1042, 256);
    const size_t pb = (size_t)Me * c.row_bytes;
    c.du = (char*)upload(c.w.data() + pb, pb);  // up allocated first: U < G
    c.dg = (char*)upload(c.w.data(), pb);
    c.dx = upload(c.x);
    c.dawq = upload(c.awq);
    c.ds1 = upload(c.s1);
    c.ds2 = upload(c.s2);
    const size_t yn = (size_t)Me * N;
    OK(hipMalloc(&c.yg, yn * 4));
    OK(hipMalloc(&c.yu, yn * 4));
    OK(hipMalloc(&c.yh, yn * 4));
    OK(hipMalloc(&c.xrot, yn * 4));
    OK(hipMalloc(&c.xi4, (size_t)(Me / 128) * N * 72));
}

static void free_case(Case& c) {
    for (void* p : {(void*)c.dg, (void*)c.du, (void*)c.dx, (void*)c.dawq, (void*)c.ds1,
                    (void*)c.ds2, (void*)c.yg, (void*)c.yu, (void*)c.yh, (void*)c.xrot,
                    (void*)c.xi4})
        OK(hipFree(p));
}

// ---------------------------------------------------------------- launches
static void gemm_set(hipFunction_t f, const char* A, const block_i4_128* X, float* Y, int M,
                     int K, int N) {
    void* args[] = {&A, &X, &Y, &M, &K, &N};
    OK(hipModuleLaunchKernel(f, N / g_geo.tn, M / g_geo.tm, 1, 32, g_geo.waves, 1, g_geo.lds,
                             nullptr, args, nullptr));
}
static void gemm_silu(const char* G, const char* U, const block_i4_128* X, float* H, int M,
                      int K, int N) {
    void* args[] = {&G, &U, &X, &H, &M, &K, &N};
    OK(hipModuleLaunchKernel(g_silu, N / g_geo.tn, 2 * M / g_geo.tm, 1, 32, g_geo.waves, 1,
                             g_geo.lds, nullptr, args, nullptr));
}
static void prod(Case& c, hipFunction_t f, const float* g, const float* u, float* xrot) {
    int K = c.Me, N = c.N;
    void* args[] = {&g, &u, &c.dawq, &c.ds1, &c.ds2, &xrot, &c.xi4, &K, &N};
    OK(hipModuleLaunchKernel(f, K / 256, N, 1, 32, 1, 1, 0, nullptr, args, nullptr));
}
static void hin(Case& c, const float* h, float* xrot) {
    int K = c.Me, N = c.N;
    void* args[] = {&h, &c.dawq, &c.ds1, &c.ds2, &xrot, &c.xi4, &K, &N};
    OK(hipModuleLaunchKernel(g_hin, K / 256, N, 1, 32, 1, 1, 0, nullptr, args, nullptr));
}

// Arm step lists; `xrot` non-null only in the oracle.
static void arm_A(Case& c, float* xrot, hipEvent_t* ev) {
    if (ev) OK(hipEventRecord(ev[0]));
    gemm_set(g_set_base, c.dg, c.dx, c.yg, c.Me, c.K, c.N);
    gemm_set(g_set_base, c.du, c.dx, c.yu, c.Me, c.K, c.N);
    if (ev) OK(hipEventRecord(ev[1]));
    prod(c, g_prod, c.yg, c.yu, xrot);
    if (ev) OK(hipEventRecord(ev[2]));
}
static void arm_C(Case& c, float* xrot, hipEvent_t* ev) {
    if (ev) OK(hipEventRecord(ev[0]));
    gemm_silu(c.dg, c.du, c.dx, c.yh, c.Me, c.K, c.N);
    if (ev) OK(hipEventRecord(ev[1]));
    hin(c, c.yh, xrot);
    if (ev) OK(hipEventRecord(ev[2]));
}

// ---------------------------------------------------------------- oracle
// CPU value of one GEMM output under the pinned fold DAG:
//   sum = fma(RN(float(sc_h) * d_h), float(C_h), sum), ascending K128 half h.
static float cpu_out(const Case& c, int r, int t) {
    const int H = c.K / 128;
    float sum = 0.0f;
    for (int h = 0; h < H; ++h) {
        const unsigned char* gp = c.w.data() + (size_t)r * c.row_bytes + (size_t)(h / 2) * 136;
        const int hh = h & 1;
        uint16_t scb;
        memcpy(&scb, gp + 4 * hh, 2);
        const float sc = (float)*(const _Float16*)&scb;
        const block_i4_128& b = c.x[(size_t)h * c.N + t];
        int C = 0;
        for (int kk = 0; kk < 128; ++kk) {
            const unsigned wb = gp[8 + 64 * hh + kk / 2];
            const int wn = (int)((kk & 1) ? (wb >> 4) : (wb & 15)) - 8;
            const unsigned xb = b.qs[kk / 2];
            int xn = (int)((kk & 1) ? (xb >> 4) : (xb & 15));
            if (xn >= 8) xn -= 16;
            C += wn * xn;
        }
        const float t1 = sc * b.d;
        sum = std::fmaf(t1, (float)C, sum);
    }
    return sum;
}

template <class T>
static size_t count_diff(const std::vector<T>& a, const std::vector<T>& b, size_t* first) {
    size_t d = 0;
    *first = (size_t)-1;
    for (size_t i = 0; i < a.size(); ++i)
        if (memcmp(&a[i], &b[i], sizeof(T))) {
            if (!d) *first = i;
            ++d;
        }
    return d;
}

static bool g_fail = false;
static void report(const char* what, size_t n, size_t diff, size_t first) {
    printf("ORACLE %s compared=%zu differing=%zu", what, n, diff);
    if (diff) printf(" first=%zu", first);
    printf("\n");
    fflush(stdout);
    if (diff) g_fail = true;
}

static void oracle_case(int Me, int K, int N, XMode xm, const char* tag) {
    Case c;
    build_case(c, Me, K, N, xm);
    const size_t yn = (size_t)Me * N, xn = (size_t)(Me / 128) * N;
    printf("CASE %s Me=%d K=%d N=%d\n", tag, Me, K, N);
    size_t f;
    // A: production pair + production producer, f32 x_rot emitted.
    OK(hipMemset(c.xi4, 0xA5, xn * 72));
    OK(hipMemset(c.xrot, 0x7f, yn * 4));
    arm_A(c, c.xrot, nullptr);
    OK(hipDeviceSynchronize());
    auto g = download(c.yg, yn), u = download(c.yu, yn), rA = download(c.xrot, yn);
    auto qA = download(c.xi4, xn);
    // CPU spot check of the production GEMM outputs (both projections).
    size_t spot = 0, spot_bad = 0;
    for (int i = 0; i < 2048; ++i) {
        const uint64_t r = xreuse_oracle::mix64(UINT64_C(0x9e3779b97f4a7c15) + i);
        const int row = (int)(r % Me), t = (int)((r >> 32) % N);
        const float eg = cpu_out(c, row, t);
        const float eu = cpu_out(c, Me + row, t);
        spot += 2;
        spot_bad += memcmp(&eg, &g[(size_t)t * Me + row], 4) != 0;
        spot_bad += memcmp(&eu, &u[(size_t)t * Me + row], 4) != 0;
    }
    report("cpu_spot_vs_base_gemm", spot, spot_bad, 0);
    // The branch object's SET entry equals the base object's.
    OK(hipMemset(c.yh, 0x7f, yn * 4));
    gemm_set(g_set_new, c.du, c.dx, c.yh, Me, K, N);
    OK(hipDeviceSynchronize());
    report("new_set_vs_base_set_up", yn, count_diff(download(c.yh, yn), u, &f), f);
    // A without x_rot gives the same block_i4_128 bytes (timing arms skip x_rot).
    OK(hipMemset(c.xi4, 0x5A, xn * 72));
    arm_A(c, nullptr, nullptr);
    OK(hipDeviceSynchronize());
    report("A_noxrot_vs_A_xi4", xn * 72, count_diff(download(c.xi4, xn), qA, &f), f);
    if (g_prod_jit) {
        OK(hipMemset(c.xi4, 0x5A, xn * 72));
        OK(hipMemset(c.xrot, 0x7f, yn * 4));
        prod(c, g_prod_jit, c.yg, c.yu, c.xrot);
        OK(hipDeviceSynchronize());
        report("jit_producer_xrot_vs_local", yn, count_diff(download(c.xrot, yn), rA, &f), f);
        report("jit_producer_xi4_vs_local", xn * 72, count_diff(download(c.xi4, xn), qA, &f), f);
    }
    // C: F1-lite silu epilogue + h-producer must equal A on the f32 rotated
    // row (every element) and on the block_i4_128 bytes.
    OK(hipMemset(c.yh, 0x7f, yn * 4));
    OK(hipMemset(c.xi4, 0x5A, xn * 72));
    OK(hipMemset(c.xrot, 0x7f, yn * 4));
    arm_C(c, c.xrot, nullptr);
    OK(hipDeviceSynchronize());
    report("C_xrot_vs_A", yn, count_diff(download(c.xrot, yn), rA, &f), f);
    report("C_xi4_vs_A", xn * 72, count_diff(download(c.xi4, xn), qA, &f), f);
    // h itself vs the CPU of SILU_MUL on the pair's g, u is not bitwise
    // meaningful (host expf); coverage of the exercised silu input range:
    auto h = download(c.yh, yn);
    size_t big = 0, zero = 0, neg = 0, nonfin = 0;
    for (size_t i = 0; i < yn; ++i) {
        big += std::fabs(g[i]) > 88.0f;
        zero += g[i] == 0.0f;
        neg += g[i] < 0.0f;
        nonfin += !std::isfinite(h[i]);
    }
    printf("COVERAGE %s |g|>88=%zu g==0=%zu g<0=%zu h_nonfinite=%zu of %zu\n", tag, big, zero,
           neg, nonfin, yn);
    // C without x_rot gives the same block_i4_128 bytes.
    OK(hipMemset(c.xi4, 0x5A, xn * 72));
    arm_C(c, nullptr, nullptr);
    OK(hipDeviceSynchronize());
    report("C_noxrot_xi4_vs_A", xn * 72, count_diff(download(c.xi4, xn), qA, &f), f);
    free_case(c);
}

static void oracle() {
    const int N = getenv("ORACLE_N") ? atoi(getenv("ORACLE_N")) : 8192;
    oracle_case(17408, 5120, N, XMode::Hashed, "full_hashed");
    oracle_case(1024, 1024, 512, XMode::SiluSpecials, "silu_specials");
    oracle_case(512, 512, 768, XMode::Hashed, "small_multitile");
    // Producer K (= Me) must be a multiple of 256; 2Me/TM = 12 (V2C) / 6 (V2B).
    oracle_case(768, 768, 1280, XMode::SiluSpecials, "odd_tiles");
    printf(g_fail ? "ORACLE_FAIL\n" : "ORACLE_PASS\n");
    if (g_fail) exit(5);
}

// ---------------------------------------------------------------- timing
static void timing(bool reverse, int N) {
    Case c;
    build_case(c, 17408, 5120, N, XMode::Hashed);
    struct ArmT {
        const char* name;
        int id;
    } arms[] = {{"A_pair", 0}, {"C_f1lite", 1}};
    const int na = 2;
    hipEvent_t ev[3];
    for (auto& e : ev) OK(hipEventCreate(&e));
    auto run = [&](int id, hipEvent_t* e) {
        if (id == 0) arm_A(c, nullptr, e);
        else arm_C(c, nullptr, e);
    };
    for (int ii = 0; ii < na; ++ii) {
        const ArmT& a = arms[reverse ? na - 1 - ii : ii];
        const double w0 = wall();
        for (int j = 0; j < 4 || wall() - w0 < 0.6; ++j) {
            run(a.id, nullptr);
            OK(hipDeviceSynchronize());
        }
        std::vector<float> tot, k1, k2;
        const double t0 = wall();
        for (int j = 0; j < 7; ++j) {
            OK(hipDeviceSynchronize());
            run(a.id, ev);
            OK(hipEventSynchronize(ev[2]));
            float m0, m1, mt;
            OK(hipEventElapsedTime(&m0, ev[0], ev[1]));
            OK(hipEventElapsedTime(&m1, ev[1], ev[2]));
            OK(hipEventElapsedTime(&mt, ev[0], ev[2]));
            k1.push_back(m0), k2.push_back(m1), tot.push_back(mt);
        }
        const double t1 = wall();
        auto med = [](std::vector<float> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        printf("RESULT arm=%s stem=%s N=%d total_ms=%.4f gemm_ms=%.4f producer_ms=%.4f "
               "begin=%.3f end=%.3f all=",
               a.name, g_geo.stem.c_str(), N, med(tot), med(k1), med(k2), t0, t1);
        for (float m : tot) printf("%.4f,", m);
        printf("\n");
        fflush(stdout);
    }
    for (auto& e : ev) OK(hipEventDestroy(e));
    free_case(c);
}

int main(int argc, char** argv) {
    const char* rv = getenv("ROCR_VISIBLE_DEVICES");
    const char* hv = getenv("HIP_VISIBLE_DEVICES");
    const char* arch = getenv("EXPECTED_ARCH");
    if (!rv || !hv || !arch || strcmp(hv, "0")) return 3;
    if (!strcmp(arch, "gfx1151") && strcmp(rv, "1")) return 3;
    if (!strcmp(arch, "gfx1100") && strcmp(rv, "0")) return 3;
    hipDeviceProp_t p{};
    OK(hipGetDeviceProperties(&p, 0));
    printf("DEVICE arch=%s name=%s mps=%d ROCR=%s HIP=%s pid=%d\n", p.gcnArchName, p.name,
           p.multiProcessorCount, rv, hv, (int)getpid());
    fflush(stdout);
    if (strncmp(p.gcnArchName, arch, strlen(arch))) return 3;
    init_modules();
    resources();
    const std::string mode = argc > 1 ? argv[1] : "resources";
    if (mode == "oracle") oracle();
    else if (mode == "time") {
        const bool rev = argc > 2 && !strcmp(argv[2], "rev");
        const int N = argc > 3 ? atoi(argv[3]) : 8192;
        timing(rev, N);
    }
    return 0;
}
