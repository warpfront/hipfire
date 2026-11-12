// gluetest: bitwise A/B of the qwen4_g942_moe_wide glue kernels against the C1 fold kernels
// (qwen4_g942_glue_moe) and the incumbent router (moe_router_softmax_top10_f32), plus timing.
// usage: gluetest <old_glue.co> <old_router.co> <new_wide.co> [trials] [--time-only]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); exit(3); } } while (0)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) { for (int i = 0; i < 4; i++) next(); }
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t u32() { return (uint32_t)(next() >> 16); }
    double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    double gauss() { double u = uni() + 1e-300, v = uni(); return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v); }
    int below(int n) { return (int)(next() % (uint64_t)n); }
};

struct Args {
    std::vector<uint8_t> b;
    void align(size_t a) { while (b.size() % a) b.push_back(0); }
    void ptr(const void* p) { align(8); const uint8_t* q = (const uint8_t*)&p; b.insert(b.end(), q, q + 8); }
    void i32(int v) { align(4); const uint8_t* q = (const uint8_t*)&v; b.insert(b.end(), q, q + 4); }
};

static void launch(hipFunction_t f, int gx, int gy, int bx, Args& a) {
    size_t sz = a.b.size();
    void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.b.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(f, gx, gy, 1, bx, 1, 1, 0, nullptr, nullptr, extra));
}

static hipFunction_t getf(hipModule_t m, const char* n) {
    hipFunction_t f; hipError_t e = hipModuleGetFunction(&f, m, n);
    if (e != hipSuccess) { fprintf(stderr, "no function %s\n", n); exit(2); }
    return f;
}

template <class T> static T* dalloc(size_t n) { T* p; CK(hipMalloc((void**)&p, n * sizeof(T))); return p; }
template <class T> static void h2d(T* d, const std::vector<T>& h) { CK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); }
template <class T> static std::vector<T> d2h(const T* d, size_t n) { std::vector<T> h(n); CK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }

static hipModule_t g_old, g_rt, g_new;
static hipFunction_t f_old_scatter, f_old_unscatter, f_old_combine, f_old_router;
static hipFunction_t f_new_scatter, f_new_unscatter, f_new_combine, f_new_router;

static long long mism = 0, cmps = 0;
template <class T> static bool same(const char* what, const std::vector<T>& a, const std::vector<T>& b, const char* ctx) {
    cmps++;
    if (a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0) return true;
    size_t i = 0; while (i < a.size() && memcmp(&a[i], &b[i], sizeof(T)) == 0) i++;
    mism++;
    if (mism <= 20) {
        uint64_t x[2] = {0, 0}, y[2] = {0, 0}; memcpy(x, &a[i], sizeof(T) < 16 ? sizeof(T) : 16); memcpy(y, &b[i], sizeof(T) < 16 ? sizeof(T) : 16);
        fprintf(stderr, "MISMATCH %s [%s] at %zu: old 0x%llx:%llx new 0x%llx:%llx\n", what, ctx, i, (unsigned long long)x[0], (unsigned long long)x[1], (unsigned long long)y[0], (unsigned long long)y[1]);
    }
    return false;
}

struct Pool {
    std::vector<float> g;  // gaussian pool
    Pool() { Rng r(99); g.resize(1 << 22); for (auto& v : g) v = (float)r.gauss(); }
    void fill(Rng& r, std::vector<float>& out, float scale) {
        size_t off = r.next() % (g.size() - 1000);
        for (size_t i = 0; i < out.size(); i++) { size_t k = off + i; if (k >= g.size()) k %= g.size(); out[i] = g[k] * scale; }
    }
};
static Pool* g_pool;
static int n_tiles_for(int slots) { return (slots + std::min(slots, 512) * 15 + 15) / 16; }

static float bf16_round_host(float v) {
    uint32_t u; memcpy(&u, &v, 4);
    if (!std::isfinite(v)) return v;
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u; memcpy(&v, &u, 4); return v;
}

// ------------------------------------------------------------------ router
static void gen_logits(Rng& r, int type, std::vector<float>& L, int tokens) {
    L.assign((size_t)tokens * 512, 0.f);
    for (int t = 0; t < tokens; t++) {
        float* row = &L[(size_t)t * 512];
        int ty = type >= 0 ? type : r.below(8);
        for (int e = 0; e < 512; e++) {
            double v = 0;
            switch (ty) {
                case 0: v = r.gauss() * 3.0; break;
                case 1: v = std::round(r.gauss() * 4.0) * 0.25; break;                 // coarse ties
                case 2: v = 0.75; break;                                               // all equal
                case 3: { static const float z[] = {0.f, -0.f, 1e-30f, -1e-30f, 5.f, 5.00001f, 5.01f, -5.f}; v = z[r.below(8)]; break; }
                case 4: v = (r.uni() - 0.5) * 2e30; break;                             // huge
                case 5: v = r.gauss() * 3.0; break;                                    // NaN/Inf injected below
                case 6: v = -100.0 + r.gauss() * 0.3; break;                           // large negative
                case 7: v = std::round(r.gauss() * 40.0) / 128.0; break;              // bf16-representable ties
            }
            row[e] = (float)v;
        }
        if (ty == 5 && r.below(2)) {
            int k = 1 + r.below(3);
            for (int j = 0; j < k; j++) {
                int e = r.below(512); int w = r.below(3);
                row[e] = w == 0 ? NAN : w == 1 ? INFINITY : -INFINITY;
            }
        }
        // values that are bf16-ties after rounding but differ in F32
        if (ty == 1 && r.below(3) == 0) for (int e = 0; e < 512; e++) { uint32_t u; memcpy(&u, &row[e], 4); u ^= (r.u32() & 0x3FFF); memcpy(&row[e], &u, 4); }
    }
}

static void test_router(int trials) {
    const int TOK[] = {8, 9, 17, 31, 32, 33, 40, 63, 64};
    float *dlog = dalloc<float>(64 * 512);
    int *dix_o = dalloc<int>(64 * 10), *dix_n = dalloc<int>(64 * 10);
    float *dw_o = dalloc<float>(64 * 10), *dw_n = dalloc<float>(64 * 10);
    unsigned long long *dfault = dalloc<unsigned long long>(2);
    Rng r(11);
    for (int it = 0; it < trials; it++) {
        const int tokens = TOK[r.below(9)];
        const int type = (it % 9 == 8) ? -1 : it % 8;
        const int norm = r.below(2);
        std::vector<float> L; gen_logits(r, type, L, tokens);
        h2d(dlog, L);
        std::vector<int> pi((size_t)tokens * 10, 0x5A5A5A5A); std::vector<float> pw((size_t)tokens * 10, 12345.f);
        h2d(dix_o, pi); h2d(dix_n, pi); h2d(dw_o, pw); h2d(dw_n, pw);
        unsigned long long zero[2] = {0, 0};
        CK(hipMemcpy(dfault, zero, 16, hipMemcpyHostToDevice));
        {
            Args a; a.ptr(dlog); a.ptr(dix_o); a.ptr(dw_o); a.i32(512); a.i32(10); a.i32(norm); a.i32(1);
            a.ptr(dfault); a.i32(7); a.i32(1000);
            launch(f_old_router, tokens, 1, 256, a);
        }
        CK(hipDeviceSynchronize());
        unsigned long long fo[2]; CK(hipMemcpy(fo, dfault, 16, hipMemcpyDeviceToHost));
        CK(hipMemcpy(dfault, zero, 16, hipMemcpyHostToDevice));
        {
            Args a; a.ptr(dlog); a.ptr(dix_n); a.ptr(dw_n); a.i32(512); a.i32(10); a.i32(norm); a.i32(1); a.i32(tokens);
            a.ptr(dfault); a.i32(7); a.i32(1000);
            launch(f_new_router, (tokens + 3) / 4, 1, 256, a);
        }
        CK(hipDeviceSynchronize());
        unsigned long long fn[2]; CK(hipMemcpy(fn, dfault, 16, hipMemcpyDeviceToHost));
        char ctx[96]; snprintf(ctx, sizeof ctx, "router it=%d tok=%d type=%d norm=%d", it, tokens, type, norm);
        same("router ids", d2h(dix_o, (size_t)tokens * 10), d2h(dix_n, (size_t)tokens * 10), ctx);
        same("router weights", d2h(dw_o, (size_t)tokens * 10), d2h(dw_n, (size_t)tokens * 10), ctx);
        std::vector<unsigned long long> va(fo, fo + 2), vb(fn, fn + 2);
        // fault: with simultaneous writers the last store wins (order unspecified), so only compare "row set" membership
        cmps++;
        if ((fo[0] == 0) != (fn[0] == 0)) { mism++; fprintf(stderr, "MISMATCH router fault presence [%s] old %llx new %llx\n", ctx, fo[0], fn[0]); }
    }
    printf("router: %d trials done\n", trials); fflush(stdout);
}

// ------------------------------------------------------------------ scatter + unscatter + combine
struct Route { int L, tokens, Pmax, ntiles; std::vector<int> topk; };

static Route gen_route(Rng& r, int tokens) {
    Route R; R.tokens = tokens; R.L = tokens * 10; R.ntiles = n_tiles_for(R.L); R.Pmax = R.ntiles * 16;
    R.topk.assign(R.L, 0);
    int mode = r.below(6);
    int hot = 8 + r.below(60);
    for (int t = 0; t < tokens; t++) {
        // ten distinct experts per token (a real router's top-10)
        int chosen[10];
        for (int k = 0; k < 10; k++) {
            for (;;) {
                int e;
                switch (mode) {
                    case 0: e = r.below(512); break;
                    case 1: e = r.below(hot); break;                      // skewed
                    case 2: e = (r.below(4) == 0) ? r.below(512) : r.below(24); break;
                    case 3: e = k; break;                                  // every token routes experts 0..9
                    case 4: e = 502 + k; break;                            // top of range
                    default: e = r.below(512); break;
                }
                bool dup = false; for (int j = 0; j < k; j++) dup |= chosen[j] == e;
                if (!dup) { chosen[k] = e; break; }
            }
            int e = chosen[k];
            if (r.below(100) == 0) { static const int bad[] = {-1, 512, 99999, -7, 0x7fffffff}; e = bad[r.below(5)]; }
            R.topk[t * 10 + k] = e;
        }
    }
    return R;
}

static void test_glue(int trials) {
    const int TOK[] = {8, 9, 10, 16, 17, 31, 32, 33, 48, 63, 64};
    const int E = 512, MI = 640, H = 2560, SHMI = 640;
    int *dtopk = dalloc<int>(1024);
    int *dcnt[2], *doff[2], *dsrt[2], *dtile[2], *dinv[2]; int4 *ddesc[2];
    const int PMAXALLOC = n_tiles_for(640) * 16;
    for (int k = 0; k < 2; k++) {
        dcnt[k] = dalloc<int>(E); doff[k] = dalloc<int>(E + 1); dsrt[k] = dalloc<int>(PMAXALLOC); dtile[k] = dalloc<int>(PMAXALLOC / 16);
        dinv[k] = dalloc<int>(1024); ddesc[k] = dalloc<int4>(PMAXALLOC / 16);
    }
    float* dY = dalloc<float>((size_t)PMAXALLOC * 2 * MI);
    float *dact[2] = {dalloc<float>((size_t)640 * MI), dalloc<float>((size_t)640 * MI)};
    float *dsg = dalloc<float>(64 * SHMI), *dsu = dalloc<float>(64 * SHMI);
    float *dsa[2] = {dalloc<float>(64 * SHMI), dalloc<float>(64 * SHMI)}, *dsel[2] = {dalloc<float>(64), dalloc<float>(64)};
    float* dgd = dalloc<float>((size_t)PMAXALLOC * H);
    float *dres[2] = {dalloc<float>(64 * H), dalloc<float>(64 * H)};
    float *dsho = dalloc<float>(64 * H), *dsc = dalloc<float>(64), *dtw = dalloc<float>(640);
    Rng r(23);
    for (int it = 0; it < trials; it++) {
        const int tokens = TOK[r.below(11)];
        Route R = gen_route(r, tokens);
        char ctx[96]; snprintf(ctx, sizeof ctx, "glue it=%d tok=%d", it, tokens);
        h2d(dtopk, R.topk);
        const int poison = 0xABABABAB;
        auto reset_scatter = [&](int k) {
            std::vector<int> p(PMAXALLOC, poison);
            h2d(dcnt[k], std::vector<int>(E, poison)); h2d(doff[k], std::vector<int>(E + 1, poison)); h2d(dsrt[k], p);
            h2d(dtile[k], std::vector<int>(PMAXALLOC / 16, poison)); h2d(dinv[k], std::vector<int>(1024, poison));
            h2d(ddesc[k], std::vector<int4>(PMAXALLOC / 16, int4{poison, poison, poison, poison}));
        };
        reset_scatter(0); reset_scatter(1);
        auto run_scatter = [&](hipFunction_t f, int k) {
            Args a; a.ptr(dtopk); a.ptr(dcnt[k]); a.ptr(doff[k]); a.ptr(dsrt[k]); a.ptr(dtile[k]); a.ptr(dinv[k]); a.ptr(ddesc[k]);
            a.i32(R.L); a.i32(E); a.i32(R.Pmax);
            launch(f, 1, 1, 1024, a);
        };
        run_scatter(f_old_scatter, 0); run_scatter(f_new_scatter, 1);
        CK(hipDeviceSynchronize());
        same("scatter counts", d2h(dcnt[0], E), d2h(dcnt[1], E), ctx);
        same("scatter offsets", d2h(doff[0], E + 1), d2h(doff[1], E + 1), ctx);
        same("scatter sorted", d2h(dsrt[0], R.Pmax), d2h(dsrt[1], R.Pmax), ctx);
        same("scatter tiles", d2h(dtile[0], R.ntiles), d2h(dtile[1], R.ntiles), ctx);
        same("scatter inverse", d2h(dinv[0], R.L), d2h(dinv[1], R.L), ctx);
        same("scatter tile_desc", d2h(ddesc[0], R.ntiles), d2h(ddesc[1], R.ntiles), ctx);

        // ---- unscatter (from the old scatter's sorted / inverse)
        std::vector<float> Y((size_t)R.Pmax * 2 * MI);
        g_pool->fill(r, Y, 4.0f); for (int k = 0; k < 2000; k++) Y[r.next() % Y.size()] *= 250.0f;
        h2d(dY, Y);
        std::vector<float> sg((size_t)tokens * SHMI), su(sg.size()), sel(tokens);
        for (auto& v : sg) v = (float)(r.gauss() * 3.0); for (auto& v : su) v = (float)(r.gauss() * 3.0); for (auto& v : sel) v = (float)(r.gauss() * 2.0);
        h2d(dsg, sg); h2d(dsu, su);
        const int grid_y = (MI + 255) / 256; const int shared_n = tokens * SHMI; const int act_blocks = (shared_n + 255) / 256;
        for (int k = 0; k < 2; k++) {
            h2d(dact[k], std::vector<float>((size_t)R.L * MI, 12345.f)); h2d(dsa[k], std::vector<float>((size_t)shared_n, 777.f)); h2d(dsel[k], sel);
        }
        {
            Args a; a.ptr(dY); a.ptr(dsrt[0]); a.ptr(dact[0]); a.i32(MI); a.i32(R.Pmax); a.i32(R.L); a.ptr(dsg); a.ptr(dsu); a.ptr(dsa[0]); a.ptr(dsel[0]);
            a.i32(tokens); a.i32(SHMI); a.i32(act_blocks);
            launch(f_old_unscatter, R.Pmax + (act_blocks + grid_y - 1) / grid_y, grid_y, 256, a);
        }
        {
            Args a; a.ptr(dY); a.ptr(dinv[0]); a.ptr(dact[1]); a.i32(MI); a.i32(R.Pmax); a.i32(R.L); a.ptr(dsg); a.ptr(dsu); a.ptr(dsa[1]); a.ptr(dsel[1]);
            a.i32(tokens); a.i32(SHMI); a.i32(act_blocks);
            launch(f_new_unscatter, R.L + (act_blocks + grid_y - 1) / grid_y, grid_y, 256, a);
        }
        CK(hipDeviceSynchronize());
        same("unscatter act", d2h(dact[0], (size_t)R.L * MI), d2h(dact[1], (size_t)R.L * MI), ctx);
        same("unscatter shared act", d2h(dsa[0], shared_n), d2h(dsa[1], shared_n), ctx);
        same("unscatter selector", d2h(dsel[0], tokens), d2h(dsel[1], tokens), ctx);

        // ---- combine
        std::vector<float> gd((size_t)R.Pmax * H), res((size_t)tokens * H), sho((size_t)tokens * H), sc(tokens), tw((size_t)R.L);
        g_pool->fill(r, gd, 5.0f); g_pool->fill(r, res, 20.0f);
        g_pool->fill(r, sho, 5.0f); for (auto& v : sc) v = (float)r.uni();
        for (auto& v : tw) v = (float)(r.uni() * 0.5);
        h2d(dgd, gd); h2d(dsho, sho); h2d(dsc, sc); h2d(dtw, tw); h2d(dres[0], res); h2d(dres[1], res);
        auto run_combine = [&](hipFunction_t f, int k) {
            Args a; a.ptr(dgd); a.ptr(dinv[0]); a.ptr(dtopk); a.ptr(dtw); a.ptr(dres[k]); a.i32(H); a.i32(R.Pmax); a.i32(tokens); a.i32(E); a.ptr(dsho); a.ptr(dsc);
            launch(f, (H + 255) / 256, tokens, 256, a);
        };
        run_combine(f_old_combine, 0); run_combine(f_new_combine, 1);
        CK(hipDeviceSynchronize());
        same("combine residual", d2h(dres[0], (size_t)tokens * H), d2h(dres[1], (size_t)tokens * H), ctx);
    }
    printf("glue: %d trials done\n", trials); fflush(stdout);
}

// ------------------------------------------------------------------ timing
static void timing() {
    const int E = 512, MI = 640, H = 2560, SHMI = 640, tokens = 32;
    Rng r(5);
    Route R = gen_route(r, tokens);
    // realistic union: mode 0 (uniform)
    for (int t = 0; t < tokens; t++) { int ch[10]; for (int k = 0; k < 10; k++) { for (;;) { int e = r.below(512); bool d = false; for (int j = 0; j < k; j++) d |= ch[j] == e; if (!d) { ch[k] = e; break; } } R.topk[t * 10 + k] = ch[k]; } }
    int *dtopk = dalloc<int>(1024); h2d(dtopk, R.topk);
    const int P = R.Pmax;
    int *dcnt = dalloc<int>(E), *doff = dalloc<int>(E + 1), *dsrt = dalloc<int>(P), *dtile = dalloc<int>(P / 16), *dinv = dalloc<int>(1024); int4* ddesc = dalloc<int4>(P / 16);
    float *dlog = dalloc<float>(64 * 512), *dw = dalloc<float>(640); int* dix = dalloc<int>(640);
    std::vector<float> L; gen_logits(r, 0, L, tokens); h2d(dlog, L);
    unsigned long long* dfault = dalloc<unsigned long long>(2); CK(hipMemset(dfault, 0, 16));
    float* dY = dalloc<float>((size_t)P * 2 * MI); CK(hipMemset(dY, 0x3c, (size_t)P * 2 * MI * 4));
    float* dact = dalloc<float>((size_t)640 * MI);
    float *dsg = dalloc<float>(64 * SHMI), *dsu = dalloc<float>(64 * SHMI), *dsa = dalloc<float>(64 * SHMI), *dsel = dalloc<float>(64);
    CK(hipMemset(dsg, 0x3c, 64 * SHMI * 4)); CK(hipMemset(dsu, 0x3c, 64 * SHMI * 4)); CK(hipMemset(dsel, 0, 256));
    float* dgd = dalloc<float>((size_t)P * H); CK(hipMemset(dgd, 0x3c, (size_t)P * H * 4));
    float *dres = dalloc<float>(64 * H), *dsho = dalloc<float>(64 * H), *dsc = dalloc<float>(64);
    CK(hipMemset(dres, 0, 64 * H * 4)); CK(hipMemset(dsho, 0, 64 * H * 4)); CK(hipMemset(dsc, 0, 256));
    h2d(dix, std::vector<int>(640, 0)); CK(hipMemset(dw, 0, 640 * 4));
    const int grid_y = (MI + 255) / 256, shared_n = tokens * SHMI, act_blocks = (shared_n + 255) / 256;
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto timeit = [&](const char* name, auto&& fn) {
        for (int i = 0; i < 20; i++) fn();
        CK(hipDeviceSynchronize());
        const int REPS = 400;
        CK(hipEventRecord(e0, nullptr));
        for (int i = 0; i < REPS; i++) fn();
        CK(hipEventRecord(e1, nullptr)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        printf("  %-34s %8.2f us/launch (back-to-back)\n", name, ms * 1000.f / REPS);
    };
    // inverse for the consumers: run old scatter once
    {
        Args a; a.ptr(dtopk); a.ptr(dcnt); a.ptr(doff); a.ptr(dsrt); a.ptr(dtile); a.ptr(dinv); a.ptr(ddesc); a.i32(R.L); a.i32(E); a.i32(P);
        launch(f_old_scatter, 1, 1, 1024, a); CK(hipDeviceSynchronize());
    }
    printf("timing at tokens=%d slots=%d grouped_rows=%d:\n", tokens, R.L, P);
    timeit("router old (256thr/token)", [&] { Args a; a.ptr(dlog); a.ptr(dix); a.ptr(dw); a.i32(512); a.i32(10); a.i32(1); a.i32(1); a.ptr(dfault); a.i32(7); a.i32(0); launch(f_old_router, tokens, 1, 256, a); });
    timeit("router new (wave64/token)", [&] { Args a; a.ptr(dlog); a.ptr(dix); a.ptr(dw); a.i32(512); a.i32(10); a.i32(1); a.i32(1); a.i32(tokens); a.ptr(dfault); a.i32(7); a.i32(0); launch(f_new_router, (tokens + 3) / 4, 1, 256, a); });
    timeit("scatter old (fold 1)", [&] { Args a; a.ptr(dtopk); a.ptr(dcnt); a.ptr(doff); a.ptr(dsrt); a.ptr(dtile); a.ptr(dinv); a.ptr(ddesc); a.i32(R.L); a.i32(E); a.i32(P); launch(f_old_scatter, 1, 1, 1024, a); });
    timeit("scatter new (mask+popc)", [&] { Args a; a.ptr(dtopk); a.ptr(dcnt); a.ptr(doff); a.ptr(dsrt); a.ptr(dtile); a.ptr(dinv); a.ptr(ddesc); a.i32(R.L); a.i32(E); a.i32(P); launch(f_new_scatter, 1, 1, 1024, a); });
    timeit("unscatter old (row-major)", [&] { Args a; a.ptr(dY); a.ptr(dsrt); a.ptr(dact); a.i32(MI); a.i32(P); a.i32(R.L); a.ptr(dsg); a.ptr(dsu); a.ptr(dsa); a.ptr(dsel); a.i32(tokens); a.i32(SHMI); a.i32(act_blocks); launch(f_old_unscatter, P + (act_blocks + grid_y - 1) / grid_y, grid_y, 256, a); });
    timeit("unscatter new (slot-major)", [&] { Args a; a.ptr(dY); a.ptr(dinv); a.ptr(dact); a.i32(MI); a.i32(P); a.i32(R.L); a.ptr(dsg); a.ptr(dsu); a.ptr(dsa); a.ptr(dsel); a.i32(tokens); a.i32(SHMI); a.i32(act_blocks); launch(f_new_unscatter, R.L + (act_blocks + grid_y - 1) / grid_y, grid_y, 256, a); });
    auto comb = [&](hipFunction_t f) { Args a; a.ptr(dgd); a.ptr(dinv); a.ptr(dtopk); a.ptr(dw); a.ptr(dres); a.i32(H); a.i32(P); a.i32(tokens); a.i32(E); a.ptr(dsho); a.ptr(dsc); launch(f, (H + 255) / 256, tokens, 256, a); };
    timeit("combine old", [&] { comb(f_old_combine); });
    timeit("combine new (batched gathers)", [&] { comb(f_new_combine); });
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: gluetest old_glue.co old_router.co new_wide.co [trials] [--time-only]\n"); return 2; }
    int trials = argc > 4 ? atoi(argv[4]) : 2000;
    bool time_only = argc > 5 && std::string(argv[5]) == "--time-only";
    CK(hipSetDevice(0));
    CK(hipModuleLoad(&g_old, argv[1])); CK(hipModuleLoad(&g_rt, argv[2])); CK(hipModuleLoad(&g_new, argv[3]));
    f_old_scatter = getf(g_old, "qwen4_g942_moe_group_scatter_desc");
    f_old_unscatter = getf(g_old, "qwen4_g942_moe_unscatter_act");
    f_old_combine = getf(g_old, "qwen4_g942_moe_combine_shared_add");
    f_old_router = getf(g_rt, "moe_router_softmax_top10_f32");
    f_new_scatter = getf(g_new, "qwen4_g942_moe_wide_group_scatter_desc");
    f_new_unscatter = getf(g_new, "qwen4_g942_moe_wide_unscatter_act");
    f_new_combine = getf(g_new, "qwen4_g942_moe_wide_combine_shared_add");
    f_new_router = getf(g_new, "qwen4_g942_moe_wide_router");
    g_pool = new Pool();
    if (!time_only) { test_router(trials); test_glue(trials / 4 + 1); }
    timing();
    printf("compared %lld buffers, %lld MISMATCHES\n", cmps, mism);
    printf(mism == 0 ? "GLUETEST PASS\n" : "GLUETEST FAIL\n");
    return mism == 0 ? 0 : 4;
}
