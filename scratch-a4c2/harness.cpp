// Standalone oracle + timing harness for the gfx1201 norm producers.
// Loads JIT-identical code objects (gen_src.py + production hipcc flags) and
// launches them with explicit geometry supplied by the driver.
//
// usage:
//   harness <family> <mode> K N seed xrot  A:<hsaco>:<kernel>:<gx>:<gy>:<block>:<shmem>  [B:...]
//   family: rms | gated | sig      mode: oracle | time | both
//   xrot: 0/1 (pass a non-null x_rot / x_out buffer and compare it too)
// Env: REPS (timing batches, default 15), BATCH (launches/batch, default 10),
//      DATA (normal | special, default normal)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); exit(2);} } while (0)

struct Spec {
    std::string path, kernel;
    unsigned gx, gy, block, shmem;
    hipModule_t mod;
    hipFunction_t fn;
};

static Spec parse_spec(const char* s) {
    Spec sp{};
    std::vector<std::string> parts;
    std::string cur;
    for (const char* p = s; *p; ++p) {
        if (*p == ':') { parts.push_back(cur); cur.clear(); } else cur.push_back(*p);
    }
    parts.push_back(cur);
    if (parts.size() != 7) { fprintf(stderr, "bad spec %s\n", s); exit(2); }
    sp.path = parts[1]; sp.kernel = parts[2];
    sp.gx = std::stoul(parts[3]); sp.gy = std::stoul(parts[4]);
    sp.block = std::stoul(parts[5]); sp.shmem = std::stoul(parts[6]);
    CK(hipModuleLoad(&sp.mod, sp.path.c_str()));
    CK(hipModuleGetFunction(&sp.fn, sp.mod, sp.kernel.c_str()));
    return sp;
}

static std::vector<float> gen_signs(uint32_t seed, int n) {
    std::vector<float> v(n);
    uint32_t st = seed;
    for (int i = 0; i < n; ++i) {
        st = (st * 1103515245u + 12345u) & 0x7fffffffu;
        v[i] = ((st >> 16) & 1) ? 1.0f : -1.0f;
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "usage: see header\n"); return 2; }
    std::string family = argv[1], mode = argv[2];
    const int K = atoi(argv[3]), N = atoi(argv[4]);
    const uint32_t seed = (uint32_t)atoi(argv[5]);
    const bool xrot = atoi(argv[6]) != 0;
    std::vector<Spec> specs;
    for (int i = 7; i < argc; ++i) specs.push_back(parse_spec(argv[i]));
    const int reps = getenv("REPS") ? atoi(getenv("REPS")) : 15;
    const int batch = getenv("BATCH") ? atoi(getenv("BATCH")) : 10;
    const std::string data = getenv("DATA") ? getenv("DATA") : "normal";

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    const size_t NK = (size_t)N * K;

    // Channel-outlier profile shared by all rows (LLM-like residual stream).
    std::vector<float> chan(K);
    for (int k = 0; k < K; ++k) chan[k] = (ud(rng) < 0.01f) ? 20.0f + 30.0f * ud(rng) : 1.0f;
    auto fill_rows = [&](std::vector<float>& v, bool second) {
        for (int n = 0; n < N; ++n) {
            float rs = std::exp2f(-6.0f + 12.0f * ud(rng));
            for (int k = 0; k < K; ++k) v[(size_t)n * K + k] = nd(rng) * rs * (second ? 1.0f : chan[k]);
        }
        if (data == "special") {
            // zero row, denormal row, huge row, row with a single spike,
            // rows with exact zeros sprinkled, negative-zero row, NaN/Inf rows.
            auto row = [&](int n) { return v.data() + (size_t)(n % N) * K; };
            std::fill(row(1), row(1) + K, 0.0f);
            for (int k = 0; k < K; ++k) row(2)[k] = nd(rng) * 1e-39f;
            for (int k = 0; k < K; ++k) row(3)[k] = nd(rng) * 1e17f;
            std::fill(row(4), row(4) + K, 0.0f); row(4)[K / 3] = 3.5f;
            for (int k = 0; k < K; k += 3) row(5)[k] = 0.0f;
            std::fill(row(6), row(6) + K, -0.0f);
            for (int k = 0; k < K; ++k) row(7)[k] = (k % 2 ? 1.0f : -1.0f) * 7.0f;
            for (int k = 0; k < K; ++k) row(8)[k] = (float)((k * 37) % 15 - 7);
            for (int k = 0; k < K; ++k) row(9)[k] = nd(rng) * 1e20f;  // x*x overflows
            row(10)[17] = INFINITY; row(11)[5] = NAN;
            for (int k = 0; k < K; ++k) row(12)[k] = nd(rng) * 60.0f;  // silu/sigmoid saturation
            for (int k = 0; k < K; ++k) row(13)[k] = -std::fabs(nd(rng)) * 100.0f;
            for (int k = 0; k < K; ++k) row(14)[k] = std::ldexp(1.0f, (k % 60) - 30) * (k % 3 ? 1.f : -1.f);
        }
    };
    std::vector<float> hx(NK), hz(NK);
    fill_rows(hx, false);
    fill_rows(hz, true);
    const int WN = (family.rfind("gated", 0) == 0) ? 128 : K;
    std::vector<float> hw(WN), ha(K);
    for (int k = 0; k < WN; ++k) hw[k] = 1.0f + 0.3f * nd(rng);
    for (int k = 0; k < K; ++k) ha[k] = std::exp2f(-2.0f + 4.0f * ud(rng));
    std::vector<float> s1 = gen_signs(42, 256), s2 = gen_signs(1042, 256);

    float *dx, *dz, *dw, *da, *ds1, *ds2;
    CK(hipMalloc(&dx, NK * 4)); CK(hipMalloc(&dz, NK * 4));
    CK(hipMalloc(&dw, WN * 4)); CK(hipMalloc(&da, K * 4));
    CK(hipMalloc(&ds1, 1024)); CK(hipMalloc(&ds2, 1024));
    CK(hipMemcpy(dx, hx.data(), NK * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dz, hz.data(), NK * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, hw.data(), WN * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(da, ha.data(), K * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(ds1, s1.data(), 1024, hipMemcpyHostToDevice));
    CK(hipMemcpy(ds2, s2.data(), 1024, hipMemcpyHostToDevice));
    const size_t i4_bytes = (size_t)(K / 128) * N * 72;
    std::vector<void*> di4(specs.size()), dxr(specs.size());
    for (size_t s = 0; s < specs.size(); ++s) {
        CK(hipMalloc(&di4[s], i4_bytes));
        CK(hipMemset(di4[s], 0xA5 + (int)s, i4_bytes));
        dxr[s] = nullptr;
        if (xrot) { CK(hipMalloc(&dxr[s], NK * 4)); CK(hipMemset(dxr[s], 0x5A + (int)s, NK * 4)); }
    }

    float eps = 1e-6f;
    int n_heads = K / 128, head_dim = 128;
    auto launch = [&](size_t s, hipStream_t st) {
        Spec& sp = specs[s];
        void* xr = dxr[s];
        void* i4 = di4[s];
        int k = K, n = N;
        std::vector<void*> p;
        if (family == "rms") {
            p = {&dx, &dw, &da, &ds1, &ds2, &xr, &i4, &k, &eps, &n};
        } else if (family == "rmsn") {
            p = {&dx, &dw, &ds1, &ds2, &xr, &i4, &k, &eps, &n};
        } else if (family == "gated") {
            p = {&dx, &dz, &dw, &da, &ds1, &ds2, &xr, &i4, &n_heads, &head_dim, &eps, &k, &n};
        } else if (family == "gatedn") {
            p = {&dx, &dz, &dw, &ds1, &ds2, &xr, &i4, &n_heads, &head_dim, &eps, &k, &n};
        } else if (family == "rmsfold") {
            p = {&dx, &dz, &dw, &da, &ds1, &ds2, &xr, &i4, &k, &eps, &n};
        } else if (family == "rmsfoldn") {
            p = {&dx, &dz, &dw, &ds1, &ds2, &xr, &i4, &k, &eps, &n};
        } else if (family == "hin") {
            p = {&dx, &da, &ds1, &ds2, &xr, &i4, &k, &n};
        } else if (family == "siluawq") {
            p = {&dx, &dz, &da, &ds1, &ds2, &xr, &i4, &k, &n};
        } else if (family == "silu") {
            p = {&dx, &dz, &ds1, &ds2, &xr, &i4, &k, &n};
        } else if (family == "rotawq") {
            p = {&dx, &da, &xr, &ds1, &ds2, &i4, &k, &n};
        } else if (family == "rot") {
            p = {&dx, &xr, &ds1, &ds2, &i4, &k, &n};
        } else if (family == "sig") {
            p = {&dx, &dz, &da, &xr, &ds1, &ds2, &i4, &k, &n};
        } else {
            fprintf(stderr, "unknown family %s\n", family.c_str()); exit(2);
        }
        CK(hipModuleLaunchKernel(sp.fn, sp.gx, sp.gy, 1, sp.block, 1, 1, sp.shmem, st, p.data(), nullptr));
    };

    int rc = 0;
    if (mode == "oracle" || mode == "both") {
        for (size_t s = 0; s < specs.size(); ++s) {
            // FOLD producers update x in place: every spec starts from the same x.
            if (family.rfind("rmsfold", 0) == 0) CK(hipMemcpy(dx, hx.data(), NK * 4, hipMemcpyHostToDevice));
            launch(s, nullptr);
        }
        CK(hipDeviceSynchronize());
        if (const char* dump = getenv("DUMP")) {
            auto wr = [&](const std::string& n, const void* p, size_t b) {
                FILE* f = fopen((std::string(dump) + n).c_str(), "wb"); fwrite(p, 1, b, f); fclose(f); };
            wr(".x", hx.data(), NK * 4); wr(".z", hz.data(), NK * 4);
            wr(".w", hw.data(), WN * 4); wr(".a", ha.data(), K * 4);
            for (size_t s = 0; s < specs.size() && xrot; ++s) {
                std::vector<float> t(NK);
                CK(hipMemcpy(t.data(), dxr[s], NK * 4, hipMemcpyDeviceToHost));
                wr(".xr" + std::to_string(s), t.data(), NK * 4);
            }
        }
        std::vector<unsigned char> ref(i4_bytes), got(i4_bytes);
        CK(hipMemcpy(ref.data(), di4[0], i4_bytes, hipMemcpyDeviceToHost));
        std::vector<unsigned char> rref, rgot;
        if (xrot) { rref.resize(NK * 4); rgot.resize(NK * 4);
            CK(hipMemcpy(rref.data(), dxr[0], NK * 4, hipMemcpyDeviceToHost)); }
        // Sanity: no untouched sentinel bytes left in the reference output.
        size_t sentinel = 0;
        for (size_t i = 0; i + 72 <= i4_bytes; i += 72) {
            bool all = true;
            for (int j = 0; j < 72; ++j) all &= (ref[i + j] == 0xA5);
            sentinel += all;
        }
        printf("ref untouched blocks: %zu\n", sentinel);
        for (size_t s = 1; s < specs.size(); ++s) {
            CK(hipMemcpy(got.data(), di4[s], i4_bytes, hipMemcpyDeviceToHost));
            size_t diff = 0, first = (size_t)-1;
            for (size_t i = 0; i < i4_bytes; ++i) if (ref[i] != got[i]) { if (!diff) first = i; ++diff; }
            printf("%s vs %s: i4 differing bytes %zu / %zu", specs[s].kernel.c_str(), specs[0].kernel.c_str(), diff, i4_bytes);
            if (diff) printf(" (first byte %zu: block %zu)", first, first / 72);
            if (diff) {
                // Distinct differing tokens (block index % N) — first 12.
                std::vector<int> toks;
                for (size_t b = 0; b < i4_bytes / 72 && toks.size() < 12; ++b) {
                    if (memcmp(&ref[b * 72], &got[b * 72], 72) == 0) continue;
                    int t = (int)(b % N);
                    if (std::find(toks.begin(), toks.end(), t) == toks.end()) toks.push_back(t);
                }
                printf(" tokens:");
                for (int t : toks) printf(" %d", t);
            }
            if (xrot) {
                CK(hipMemcpy(rgot.data(), dxr[s], NK * 4, hipMemcpyDeviceToHost));
                size_t rd = 0;
                for (size_t i = 0; i < NK * 4; ++i) rd += rref[i] != rgot[i];
                printf("; x_rot differing bytes %zu / %zu", rd, NK * 4);
                diff += rd;
            }
            printf("  %s\n", diff ? "ORACLE_FAIL" : "ORACLE_PASS");
            if (diff) rc = 1;
        }
        // FNV-1a digest of the reference output (proves nontrivial content).
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < i4_bytes; ++i) { h ^= ref[i]; h *= 1099511628211ull; }
        printf("ref i4 fnv64 %016llx\n", (unsigned long long)h);
    }
    if (mode == "time" || mode == "both") {
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        for (int w = 0; w < 3; ++w) for (size_t s = 0; s < specs.size(); ++s) launch(s, nullptr);
        CK(hipDeviceSynchronize());
        std::vector<std::vector<float>> ms(specs.size());
        std::vector<float> cp;
        void* cdst; CK(hipMalloc(&cdst, NK * 4));
        for (int r = 0; r < reps; ++r) {
            for (size_t s = 0; s < specs.size(); ++s) {
                CK(hipEventRecord(e0, nullptr));
                for (int b = 0; b < batch; ++b) launch(s, nullptr);
                CK(hipEventRecord(e1, nullptr));
                CK(hipEventSynchronize(e1));
                float t; CK(hipEventElapsedTime(&t, e0, e1));
                ms[s].push_back(t / batch);
            }
            CK(hipEventRecord(e0, nullptr));
            for (int b = 0; b < batch; ++b) CK(hipMemcpyAsync(cdst, dx, NK * 4, hipMemcpyDeviceToDevice, nullptr));
            CK(hipEventRecord(e1, nullptr));
            CK(hipEventSynchronize(e1));
            float t; CK(hipEventElapsedTime(&t, e0, e1));
            cp.push_back(t / batch);
        }
        auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        auto mn = [](std::vector<float> v) { return *std::min_element(v.begin(), v.end()); };
        const bool one_stream = family.rfind("rms", 0) == 0 || family == "hin" || family.rfind("rot", 0) == 0;
        double in_streams = one_stream ? 1.0 : 2.0;
        double logical = in_streams * NK * 4.0 + (double)i4_bytes;
        float cpm = med(cp);
        printf("copy D2D %.1f MB: median %.4f ms = %.1f GB/s (r+w)\n", NK * 4 / 1e6, cpm, 2.0 * NK * 4 / cpm / 1e6);
        auto q = [](std::vector<float> v, double f) { std::sort(v.begin(), v.end()); return v[(size_t)(f * (v.size() - 1))]; };
        for (size_t s = 0; s < specs.size(); ++s) {
            float m = med(ms[s]);
            printf("#%zu %-22.22s median %.4f ms  p25 %.4f  p75 %.4f  min %.4f  logical %.1f GB/s\n", s,
                   specs[s].path.c_str(), m, q(ms[s], 0.25), q(ms[s], 0.75), mn(ms[s]), logical / m / 1e6);
        }
    }
    return rc;
}
