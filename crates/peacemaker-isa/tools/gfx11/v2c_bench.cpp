// SPDX-License-Identifier: Apache-2.0
// Standalone gfx1100 harness: builder-emitted V2C-equivalent SET GEMM vs the
// shipped hipcc V2C SET (`gemm_mq4g256v2_residual_iu4_v2c_set_gfx11`).
//
//   v2c_bench oracle <v2c.hsaco> <pm.hsaco> <W.bin> <Xq.bin> <y_ref.bin> [ref_rows]
//   v2c_bench time   <v2c.hsaco> <pm.hsaco> <W.bin> <Xq.bin> <seconds> <order, e.g. ABBABAAB>
//
// Shape M=17408, K=5120, N=8192 (override M/K/N with V2C_M/V2C_K/V2C_N).
// W: MQ4G256V2 rows (M x K/256 x 136 B); Xq: block_i4_128[h*N + t].
// y_ref: f32 [ref_rows][M] reference outputs of tokens 0..ref_rows-1 (the
// fixture's f64-equivalent A4 product); tokens repeat with period ref_rows.
// Oracle: whole Y buffers (plus 64-float guards) compared bytewise under
// three poison patterns, repeat determinism, token-period equality, and a
// reference tolerance check. Time: each arm launches its kernel back to back
// for >= seconds (HIP events per 10-launch batch) in the given order while a
// 20 Hz sysfs sampler records the card's power and shader clock.
// Requires exactly one visible HIP device: gfx1100 at PCI bus 0x66.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <glob.h>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); exit(1);} } while (0)

static std::vector<unsigned char> slurp(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "read %s failed\n", path); exit(1); }
    fclose(f);
    return v;
}
static int env_int(const char* k, int d) { const char* v = getenv(k); return v ? atoi(v) : d; }

struct Kernel {
    hipModule_t mod; hipFunction_t fn; unsigned bx, by;
    Kernel(const char* path, const char* sym, unsigned bx_, unsigned by_) : bx(bx_), by(by_) {
        CK(hipModuleLoad(&mod, path)); CK(hipModuleGetFunction(&fn, mod, sym));
        int v = 0, s = 0, p = 0;
        CK(hipFuncGetAttribute(&v, HIP_FUNC_ATTRIBUTE_NUM_REGS, fn));
        CK(hipFuncGetAttribute(&p, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
        fprintf(stderr, "loaded %s:%s vgpr=%d private=%d block=%ux%u\n", path, sym, v, p, bx, by);
        (void)s;
    }
    void run(hipStream_t st, void* A, void* X, void* Y, int M, int K, int N) const {
        void* args[] = {&A, &X, &Y, &M, &K, &N};
        CK(hipModuleLaunchKernel(fn, N / 128, M / 128, 1, bx, by, 1, 32768, st, args, nullptr));
    }
};

// Card telemetry by PCI address (hwmon under the device).
struct Sampler {
    std::string power, freq;
    std::atomic<bool> stop{false};
    std::vector<double> watts, mhz;
    std::thread th;
    Sampler() {
        glob_t g;
        if (glob("/sys/bus/pci/devices/0000:66:00.0/hwmon/hwmon*", 0, nullptr, &g) == 0 && g.gl_pathc) {
            std::string h = g.gl_pathv[0];
            power = h + "/power1_average"; freq = h + "/freq1_input";
            FILE* f = fopen(power.c_str(), "r"); if (!f) power = h + "/power1_input"; else fclose(f);
        }
        globfree(&g);
    }
    static double rd(const std::string& p) { FILE* f = fopen(p.c_str(), "r"); if (!f) return NAN; double v = NAN; if (fscanf(f, "%lf", &v) != 1) v = NAN; fclose(f); return v; }
    void start() {
        stop = false; watts.clear(); mhz.clear();
        th = std::thread([this] { while (!stop) { watts.push_back(rd(power) / 1e6); mhz.push_back(rd(freq) / 1e6);
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); } });
    }
    void finish() { stop = true; th.join(); }
};
static double median(std::vector<double> v) { if (v.empty()) return NAN; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: see header\n"); return 2; }
    const std::string mode = argv[1];
    int count = 0; CK(hipGetDeviceCount(&count));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    fprintf(stderr, "devices=%d arch=%s pci=%04x:%02x:%02x name=%s\n", count, prop.gcnArchName, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, prop.name);
    if (count != 1 || strncmp(prop.gcnArchName, "gfx1100", 7) != 0 || prop.pciBusID != 0x66) { fprintf(stderr, "device assertion failed\n"); return 2; }
    const int M = env_int("V2C_M", 17408), K = env_int("V2C_K", 5120), N = env_int("V2C_N", 8192);
    Kernel A(argv[2], "gemm_mq4g256v2_residual_iu4_v2c_set_gfx11", 32, 8);
    Kernel B(argv[3], "gemm_mq4g256v2_residual_iu4_pm_set_gfx1100", 256, 1);
    auto w = slurp(argv[4]), x = slurp(argv[5]);
    if (w.size() != (size_t)M * (K / 256) * 136 || x.size() != (size_t)(K / 128) * N * 72) { fprintf(stderr, "input sizes do not match M/K/N\n"); return 2; }
    void *dW, *dX; CK(hipMalloc(&dW, w.size())); CK(hipMalloc(&dX, x.size()));
    CK(hipMemcpy(dW, w.data(), w.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dX, x.data(), x.size(), hipMemcpyHostToDevice));
    const size_t ny = (size_t)M * N, guard = 64, bytes = (ny + guard) * 4;
    hipStream_t st; CK(hipStreamCreate(&st));
    if (mode == "oracle") {
        void *YA, *YB; CK(hipMalloc(&YA, bytes)); CK(hipMalloc(&YB, bytes));
        std::vector<unsigned char> ha(bytes), hb(bytes), first(bytes);
        const unsigned char pats[3] = {0xA5, 0x00, 0xFF};
        bool all_equal = true, repeat_equal = true;
        for (int i = 0; i < 3; ++i) {
            CK(hipMemsetAsync(YA, pats[i], bytes, st)); CK(hipMemsetAsync(YB, pats[i], bytes, st));
            A.run(st, dW, dX, YA, M, K, N); B.run(st, dW, dX, YB, M, K, N);
            CK(hipStreamSynchronize(st));
            CK(hipMemcpy(ha.data(), YA, bytes, hipMemcpyDeviceToHost)); CK(hipMemcpy(hb.data(), YB, bytes, hipMemcpyDeviceToHost));
            size_t diff = 0, first_diff = SIZE_MAX;
            for (size_t k = 0; k < bytes; ++k) if (ha[k] != hb[k]) { if (first_diff == SIZE_MAX) first_diff = k; ++diff; }
            bool guards = true;
            for (size_t k = ny * 4; k < bytes; ++k) guards &= ha[k] == pats[i] && hb[k] == pats[i];
            // Output bytes independent of the poison: every element written.
            if (i == 0) first = hb; else repeat_equal &= memcmp(first.data(), hb.data(), ny * 4) == 0;
            printf("ORACLE poison=%02x whole_buffer_equal=%s differing_bytes=%zu first_diff_float=%zd guards_untouched=%s\n",
                   pats[i], diff == 0 ? "true" : "false", diff, first_diff == SIZE_MAX ? (ssize_t)-1 : (ssize_t)(first_diff / 4), guards ? "true" : "false");
            all_equal &= diff == 0 && guards;
        }
        // Poison-independence (full coverage) and determinism of the builder output.
        printf("ORACLE builder_output_poison_independent_and_repeatable=%s\n", repeat_equal ? "true" : "false");
        const float* y = (const float*)hb.data();
        size_t finite = 0, nonzero = 0;
        for (size_t k = 0; k < ny; ++k) { finite += std::isfinite(y[k]); nonzero += y[k] != 0.0f; }
        const int period = argc > 7 ? atoi(argv[7]) : 32;
        bool periodic = true;
        for (int t = period; t < N && periodic; ++t) periodic = memcmp(y + (size_t)t * M, y + (size_t)(t % period) * M, (size_t)M * 4) == 0;
        auto ref = slurp(argv[6]);
        double max_abs = 0, max_ref = 0; size_t nref = ref.size() / 4;
        const float* r = (const float*)ref.data();
        for (size_t k = 0; k < nref && k < ny; ++k) { max_abs = std::max(max_abs, (double)fabsf(y[k] - r[k])); max_ref = std::max(max_ref, (double)fabsf(r[k])); }
        printf("ORACLE floats=%zu finite=%zu nonzero=%zu token_period_%d_bitwise=%s ref_rows=%zu max_abs_err=%.3e max_abs_ref=%.3e rel=%.3e\n",
               ny, finite, nonzero, period, periodic ? "true" : "false", nref / M, max_abs, max_ref, max_abs / max_ref);
        const bool ok = all_equal && repeat_equal && periodic && finite == ny && max_abs / max_ref < 1e-5;
        printf("ORACLE verdict=%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 3;
    }
    if (mode == "time") {
        const double secs = atof(argv[6]);
        const std::string order = argc > 7 ? argv[7] : "ABBABAAB";
        void* Y; CK(hipMalloc(&Y, bytes));
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        const int batch = 10;
        auto run_batch = [&](const Kernel& k) {
            CK(hipEventRecord(e0, st));
            for (int i = 0; i < batch; ++i) k.run(st, dW, dX, Y, M, K, N);
            CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); return (double)ms / batch;
        };
        double heat = 0; while (heat < 3000.0) heat += run_batch(A) * batch;   // 3 s preheat on the control
        Sampler s;
        const double ops = 2.0 * M * (double)K * N;
        printf("TIME shape M=%d K=%d N=%d ops=%.6e order=%s arm_seconds=%.1f\n", M, K, N, ops, order.c_str(), secs);
        std::vector<double> medA, medB;
        for (char arm : order) {
            const Kernel& k = arm == 'A' ? A : B;
            std::vector<double> t; double elapsed = 0;
            s.start();
            while (elapsed < secs * 1000.0) { double ms = run_batch(k); t.push_back(ms); elapsed += ms * batch; }
            s.finish();
            const double m = median(t);
            (arm == 'A' ? medA : medB).push_back(m);
            printf("ARM %c %s launches=%zu seconds=%.2f median_ms=%.4f min_ms=%.4f tops=%.2f power_w=%.1f sclk_mhz=%.0f samples=%zu\n",
                   arm, arm == 'A' ? "v2c_hipcc" : "pm_builder", t.size() * batch, elapsed / 1000.0, m,
                   *std::min_element(t.begin(), t.end()), ops / (m * 1e-3) / 1e12, median(s.watts), median(s.mhz), s.watts.size());
            fflush(stdout);
        }
        const double a = median(medA), b = median(medB);
        printf("TIME v2c_median_ms=%.4f pm_median_ms=%.4f pm_over_v2c=%.4f\n", a, b, b / a);
        return 0;
    }
    fprintf(stderr, "unknown mode\n");
    return 2;
}
