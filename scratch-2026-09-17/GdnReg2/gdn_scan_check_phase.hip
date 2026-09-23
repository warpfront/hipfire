// Throwaway standalone harness for the GDN register-scan slice K.
// Compiles the serial fast kernel + the candidate side by side, compares
// outputs/state, reports metadata and HIP-event timing. NOT committed.
#define HIPFIRE_GDN_DPP_REDUCE 1
#define HIPFIRE_GDN_PREFETCH 1
#include "gated_delta_net_q8_fast.hip"
#include "gdn_register_scan_q8.gfx1201.hip"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#define CK(x) do { auto e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    exit(1); } } while (0)

static unsigned long long lcg = 0x123456789abcdefULL;
static float frand(float lo, float hi) {
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = (double)(lcg >> 11) * (1.0 / 9007199254740992.0);
    return (float)(lo + (hi - lo) * u);
}

static std::vector<float> host_v;

struct Bufs {
    float *q, *k, *v, *gate, *beta, *scales, *out;
    signed char *sq;
    __half *ef;
#ifdef GDN_RS_DEBUG
    float *debug_s;
#endif
    size_t n, h;
};

static void alloc_bufs(Bufs& b, int N, int H) {
    b.n = N; b.h = H;
    CK(hipMalloc(&b.q, (size_t)N*H*128*sizeof(float)));
    CK(hipMalloc(&b.k, (size_t)N*H*128*sizeof(float)));
    CK(hipMalloc(&b.v, (size_t)N*H*128*sizeof(float)));
    CK(hipMalloc(&b.gate, (size_t)N*H*sizeof(float)));
    CK(hipMalloc(&b.beta, (size_t)N*H*sizeof(float)));
    CK(hipMalloc(&b.scales, (size_t)H*128*sizeof(float)));
    CK(hipMalloc(&b.out, (size_t)N*H*128*sizeof(float)));
    CK(hipMalloc(&b.sq, (size_t)H*128*128));
    CK(hipMalloc(&b.ef, (size_t)H*128*128*sizeof(__half)));
#ifdef GDN_RS_DEBUG
    CK(hipMalloc(&b.debug_s,
        (size_t)((N + 63) / 64)*8*H*128*128*sizeof(float)));
#endif
}

static void fill_bufs(const Bufs& b, int mode, unsigned long long seed) {
    int N = (int)b.n, H = (int)b.h;
    lcg = seed;
    std::vector<float> q((size_t)N*H*128), k((size_t)N*H*128), v((size_t)N*H*128);
    std::vector<float> gate((size_t)N*H), beta((size_t)N*H);
    for (int t = 0; t < N; t++) for (int hh = 0; hh < H; hh++) {
        if (mode == 7) { gate[(size_t)t*H+hh] = 0.0f; beta[(size_t)t*H+hh] = 1.0f; }
        else if (mode == 8) { gate[(size_t)t*H+hh] = 0.5f; beta[(size_t)t*H+hh] = 0.0f; }
        else if (mode == 6) { gate[(size_t)t*H+hh] = frand(-3.0f, -1.0f); beta[(size_t)t*H+hh] = frand(0.2f, 1.0f); }
        else if (mode == 2) { gate[(size_t)t*H+hh] = 0.0f; beta[(size_t)t*H+hh] = 1.0f; }
        else if (mode == 1) { gate[(size_t)t*H+hh] = 0.0f; beta[(size_t)t*H+hh] = 0.0f; }
        else { gate[(size_t)t*H+hh] = frand(-0.5f, 0.2f); beta[(size_t)t*H+hh] = frand(0.2f, 1.0f); }
        for (int d = 0; d < 128; d++) {
            size_t o = ((size_t)t*H+hh)*128+d;
            if (mode == 7) {
                q[o] = (d == (t % 128)) ? 1.0f : 0.0f;
                k[o] = (d == (t % 128)) ? 1.0f : 0.0f;
                v[o] = 0.0f;
            } else if (mode == 2) {
                q[o] = (d == (t % 128)) ? 1.0f : 0.0f;
                k[o] = (d == (t % 128)) ? 1.0f : 0.0f;
                v[o] = (float)((int)(frand(0, 5)) - 2);
            } else {
                q[o] = frand(-0.2f, 0.2f); k[o] = frand(-0.2f, 0.2f);
                v[o] = frand(-0.5f, 0.5f);
            }
        }
    }
    CK(hipMemcpy(b.q, q.data(), q.size()*4, hipMemcpyHostToDevice));
    CK(hipMemcpy(b.k, k.data(), k.size()*4, hipMemcpyHostToDevice));
    CK(hipMemcpy(b.v, v.data(), v.size()*4, hipMemcpyHostToDevice));
    CK(hipMemcpy(b.gate, gate.data(), gate.size()*4, hipMemcpyHostToDevice));
    host_v = v;
    CK(hipMemcpy(b.beta, beta.data(), beta.size()*4, hipMemcpyHostToDevice));
    // Q8 state + scales + EF
    std::vector<signed char> sq((size_t)H*128*128);
    std::vector<float> sc((size_t)H*128);
    std::vector<unsigned short> ef((size_t)H*128*128);
    for (size_t i = 0; i < sq.size(); i++) sq[i] = (signed char)(int)frand(-64, 64);
    for (size_t i = 0; i < sc.size(); i++) sc[i] = frand(0.001f, 0.02f);
    for (size_t i = 0; i < ef.size(); i++) {
        float f = frand(-1e-4f, 1e-4f);
        __half hh = (__half)f;
        unsigned short bits; memcpy(&bits, &hh, 2);
        ef[i] = bits;
    }
    if (mode == 7) { // exact-cancellation: small-int state, unit scale, no EF
        for (size_t i = 0; i < sq.size(); i++) sq[i] = (signed char)((int)(frand(0, 5)) - 2);
        for (size_t i = 0; i < sc.size(); i++) sc[i] = 1.0f;
        memset(ef.data(), 0, ef.size()*2);
    } else if (mode == 2 || mode == 5) { // one-hot / zero-state-random: zero state
        memset(sq.data(), 0, sq.size());
        for (size_t i = 0; i < sc.size(); i++) sc[i] = 1.0f;
        memset(ef.data(), 0, ef.size()*2);
    }
    CK(hipMemcpy(b.sq, sq.data(), sq.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(b.scales, sc.data(), sc.size()*4, hipMemcpyHostToDevice));
    CK(hipMemcpy(b.ef, ef.data(), ef.size()*2, hipMemcpyHostToDevice));
    CK(hipMemset(b.out, 0, (size_t)N*H*128*4));
#ifdef GDN_RS_DEBUG
    CK(hipMemset(b.debug_s, 0,
        (size_t)((N + 63) / 64)*8*H*128*128*sizeof(float)));
#endif
}

static void run_serial(const Bufs& b) {
    int nt = (int)b.n, nh = (int)b.h, hd = 128, fr = 0;
    hipLaunchKernelGGL(gated_delta_net_q8_fast, dim3(nh, 32, 1), dim3(32, 1, 1), 0, 0,
        b.q, b.k, b.v, b.gate, b.beta, b.sq, b.scales, b.out, nt, nh, hd, (unsigned)fr, b.ef);
}

static void run_cand(const Bufs& b) {
    int nt = (int)b.n, nh = (int)b.h, hd = 128, fr = 0;
    hipLaunchKernelGGL(gated_delta_net_q8_register_scan_gfx1201, dim3(nh, 2, 1),
        dim3(256, 1, 1), 0, 0,
        b.q, b.k, b.v, b.gate, b.beta, b.sq, b.scales, b.out, nt, nh, hd,
        (unsigned)fr, b.ef
#ifdef GDN_RS_DEBUG
        , b.debug_s
#endif
        );
}

static void download(const Bufs& b, std::vector<float>& out, std::vector<signed char>& sq,
                     std::vector<float>& sc, std::vector<unsigned short>& ef) {
    out.resize((size_t)b.n*b.h*128); sq.resize((size_t)b.h*128*128);
    sc.resize((size_t)b.h*128); ef.resize((size_t)b.h*128*128);
    CK(hipMemcpy(out.data(), b.out, out.size()*4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(sq.data(), b.sq, sq.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(sc.data(), b.scales, sc.size()*4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ef.data(), b.ef, ef.size()*2, hipMemcpyDeviceToHost));
}

static void do_meta() {
    hipFuncAttributes a, c;
    CK(hipFuncGetAttributes(&a, (const void*)gated_delta_net_q8_fast));
    CK(hipFuncGetAttributes(&c, (const void*)gated_delta_net_q8_register_scan_gfx1201));
    int occS = 0, occC = 0;
    CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occS, gated_delta_net_q8_fast, 32, 0));
    CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occC,
        gated_delta_net_q8_register_scan_gfx1201, 256, 0));
    printf("serial: vgpr=%d sgpr=%d lds=%zu const=%zu local=%zu occ=%d\n",
        a.numRegs, 0, a.sharedSizeBytes, a.constSizeBytes, a.localSizeBytes, occS);
    printf("cand:   vgpr=%d sgpr=%d lds=%zu const=%zu local=%zu occ=%d\n",
        c.numRegs, 0, c.sharedSizeBytes, c.constSizeBytes, c.localSizeBytes, occC);
}

static int do_check(int N, int H, int mode, unsigned long long seed) {
    Bufs b; alloc_bufs(b, N, H);
    fill_bufs(b, mode, seed);
    // reference: serial
    run_serial(b); CK(hipDeviceSynchronize());
    std::vector<float> oS; std::vector<signed char> qS;
    std::vector<float> sS; std::vector<unsigned short> eS;
    download(b, oS, qS, sS, eS);
    // candidate run 1
    fill_bufs(b, mode, seed);
    run_cand(b); CK(hipDeviceSynchronize());
    std::vector<float> oC; std::vector<signed char> qC;
    std::vector<float> sC; std::vector<unsigned short> eC;
    download(b, oC, qC, sC, eC);
    // candidate run 2 (determinism)
    fill_bufs(b, mode, seed);
    run_cand(b); CK(hipDeviceSynchronize());
    std::vector<float> oC2; std::vector<signed char> qC2;
    std::vector<float> sC2; std::vector<unsigned short> eC2;
    download(b, oC2, qC2, sC2, eC2);

    double maxabs = 0, sse = 0, ref = 0;
    float omax = 0;
    std::vector<double> chunkmax((N + 63) / 64, 0.0);
    for (size_t i = 0; i < oS.size(); i++) {
        double d = fabs((double)oS[i] - oC[i]);
        maxabs = std::max(maxabs, d); sse += d*d; ref += (double)oS[i]*oS[i];
        omax = std::max(omax, (float)fabs(oS[i]));
        size_t tok = (i / 128) % (size_t)N;
        chunkmax[tok / 64] = std::max(chunkmax[tok / 64], d);
    }
    double smax = 0, sref = 0, smax0 = 0, smax1 = 0;
    for (size_t i = 0; i < qS.size(); i++) {
        size_t row = i / 128, key = i % 128;
        double a = (double)qS[i] * sS[row], b = (double)qC[i] * sC[row];
        double d = fabs(a - b);
        smax = std::max(smax, d); sref = std::max(sref, fabs(a));
        if (key < 64) smax0 = std::max(smax0, d); else smax1 = std::max(smax1, d);
    }
    printf("deqhalves: %.4g %.4g\n", smax0, smax1);
    printf("chunkmax:");
    for (double c : chunkmax) printf(" %.3g", c);
    printf("\n");
    double nrmse = sqrt(sse / oS.size()) / (sqrt(ref / oS.size()) + 1e-9);
    size_t code_mm = 0;
    for (size_t i = 0; i < qS.size(); i++) if (qS[i] != qC[i]) code_mm++;
    double sc_rel = 0;
    for (size_t i = 0; i < sS.size(); i++)
        sc_rel = std::max(sc_rel, fabs((double)sS[i]-sC[i]) / (fabs((double)sS[i])+1e-9));
    bool det = (oC == oC2) && (qC == qC2) && (sC == sC2) && (eC == eC2);
    bool finite = true;
    for (float x : oC) if (!std::isfinite(x)) finite = false;
    printf("N=%d H=%d mode=%d: maxabs=%.4g nrmse=%.4g omax=%.4g codes_mm=%zu/%zu sc_rel=%.3g det=%d finite=%d deq=%.4g deqref=%.4g\n",
        N, H, mode, maxabs, nrmse, omax, code_mm, qS.size(), sc_rel, (int)det, (int)finite, smax, sref);
    if (mode == 2) {
        bool exact = (maxabs == 0.0) && (code_mm == 0);
        printf("onehot_bitexact=%d\n", (int)exact);
        return (exact && det && finite) ? 0 : 1;
    }
    int rc = (det && finite && nrmse <= 0.002 && maxabs <= 0.01*std::max(1.0f, omax)) ? 0 : 1;
    return rc;
}

#ifdef GDN_RS_DEBUG
static int do_phase(int N, int H, unsigned long long seed) {
    Bufs b; alloc_bufs(b, N, H);
    fill_bufs(b, 2, seed);
    run_cand(b); CK(hipDeviceSynchronize());
    const int chunks = (N + 63) / 64;
    std::vector<float> dump((size_t)chunks*8*H*128*128);
    CK(hipMemcpy(dump.data(), b.debug_s, dump.size()*sizeof(float),
                 hipMemcpyDeviceToHost));
    std::vector<float> expected((size_t)H*128*128, 0.0f);
    bool exact = true;
    for (int chunk = 0; chunk < chunks; chunk++) {
        for (int phase = 0; phase < 8; phase++) {
            if (phase == 7) {
                const int begin = chunk * 64;
                const int end = std::min(begin + 64, N);
                for (int t = begin; t < end; t++) {
                    const int key = t % 128;
                    for (int hh = 0; hh < H; hh++)
                        for (int row = 0; row < 128; row++)
                            expected[((size_t)hh*128 + row)*128 + key] =
                                host_v[((size_t)t*H + hh)*128 + row];
                }
            }
            double max0 = 0.0, max1 = 0.0;
            size_t mismatches = 0;
            const size_t base = ((size_t)chunk*8 + phase)*H*128*128;
            for (size_t i = 0; i < expected.size(); i++) {
                const double d = fabs((double)dump[base + i] - expected[i]);
                if ((i % 128) < 64) max0 = std::max(max0, d);
                else max1 = std::max(max1, d);
                if (d != 0.0) mismatches++;
            }
            printf("chunk=%d phase=%d S_max_halves=%.4g,%.4g mismatches=%zu\n",
                   chunk, phase, max0, max1, mismatches);
            exact = exact && mismatches == 0;
        }
    }
    printf("phase_dump_mode2_exact=%d\n", (int)exact);
    return exact ? 0 : 1;
}
#endif

static void do_time(int N, int H, int iters) {
    Bufs b; alloc_bufs(b, N, H);
    fill_bufs(b, 0, 12345);
    hipEvent_t a, evb; CK(hipEventCreate(&a)); CK(hipEventCreate(&evb));
    for (int i = 0; i < 3; i++) { run_serial(b); } CK(hipDeviceSynchronize());
    for (int i = 0; i < 3; i++) { run_cand(b); } CK(hipDeviceSynchronize());
    std::vector<double> usS, usC;
    for (int r = 0; r < 7; r++) {
        CK(hipEventRecord(a));
        for (int j = 0; j < iters; j++) run_serial(b);
        CK(hipEventRecord(evb)); CK(hipEventSynchronize(evb));
        float ms; CK(hipEventElapsedTime(&ms, a, evb)); usS.push_back(ms*1000/iters);
        CK(hipEventRecord(a));
        for (int j = 0; j < iters; j++) run_cand(b);
        CK(hipEventRecord(evb)); CK(hipEventSynchronize(evb));
        CK(hipEventElapsedTime(&ms, a, evb)); usC.push_back(ms*1000/iters);
    }
    std::sort(usS.begin(), usS.end()); std::sort(usC.begin(), usC.end());
    printf("serial med=%.2f min=%.2f max=%.2f | cand med=%.2f min=%.2f max=%.2f | ratio=%.4f\n",
        usS[3], usS[0], usS[6], usC[3], usC[0], usC[6], usC[3]/usS[3]);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: meta | check N H mode seed | time N H iters | phase N H seed\n");
        return 2;
    }
    if (!strcmp(argv[1], "meta")) { do_meta(); return 0; }
    if (!strcmp(argv[1], "check"))
        return do_check(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), strtoull(argv[5], 0, 0));
    if (!strcmp(argv[1], "time")) { do_time(atoi(argv[2]), atoi(argv[3]), atoi(argv[4])); return 0; }
#ifdef GDN_RS_DEBUG
    if (!strcmp(argv[1], "phase"))
        return do_phase(atoi(argv[2]), atoi(argv[3]), strtoull(argv[4], 0, 0));
#endif
    return 2;
}
