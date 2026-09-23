#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

static constexpr int NH = 24;
static constexpr int NKV = 4;
static constexpr int HD = 256;
static constexpr int BATCH = 512;
static constexpr int ROW_BYTES = 1032;
static constexpr int MAX_CTX = 32768;
static constexpr int WARMUPS = 5;
static constexpr int SAMPLES = 25;
static constexpr int LENGTHS[] = {96, 257, 1536, 3072, 6144, 12288, 32768};
static constexpr float SCALE = 1.0f / 16.0f;

static void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::cerr << what << ": " << hipGetErrorString(e) << "\n";
        std::exit(1);
    }
}

struct Rng {
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    uint64_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state *= 0x2545f4914f6cdd1dULL;
        return state;
    }
};

struct Arm {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    int grid_x = 0;
    int block_x = 0;
    unsigned lds = 0;
};

static double sample(
    const Arm& arm,
    void* q, void* q8, void* k, void* v, void* out, void* positions) {
    int nh = NH, nkv = NKV, hd = HD, batch = BATCH;
    float scale = SCALE;
    void* args[] = {&q, &q8, &k, &v, &out, &positions,
                    &nh, &nkv, &hd, &batch, &scale};
    hipEvent_t start = nullptr, stop = nullptr;
    check(hipEventCreate(&start), "hipEventCreate start");
    check(hipEventCreate(&stop), "hipEventCreate stop");
    check(hipEventRecord(start, nullptr), "hipEventRecord start");
    check(hipModuleLaunchKernel(
              arm.function, arm.grid_x, 4, 1, arm.block_x, 1, 1, arm.lds,
              nullptr, args, nullptr),
          "hipModuleLaunchKernel");
    check(hipEventRecord(stop, nullptr), "hipEventRecord stop");
    check(hipEventSynchronize(stop), "hipEventSynchronize");
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime");
    check(hipEventDestroy(start), "hipEventDestroy start");
    check(hipEventDestroy(stop), "hipEventDestroy stop");
    return static_cast<double>(ms) * 1000.0;
}

struct Stats {
    double median;
    double p95;
    std::vector<double> sorted;
};

static Stats stats(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return {values[values.size() / 2],
            values[(values.size() * 95 + 99) / 100 - 1],
            std::move(values)};
}

static void print_samples(const std::vector<double>& v) {
    std::cout << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << v[i];
    }
    std::cout << ']';
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: pinned_body_timing BASE_HSACO CANDIDATE_HSACO\n";
        return 2;
    }
    check(hipSetDevice(0), "hipSetDevice");
    Arm base, candidate;
    check(hipModuleLoad(&base.module, argv[1]), "load baseline");
    check(hipModuleLoad(&candidate.module, argv[2]), "load candidate");
    const char* symbol = "attention_fp8_e4m3_fa2_gqa_packet_gfx1201";
    check(hipModuleGetFunction(&base.function, base.module, symbol), "baseline symbol");
    check(hipModuleGetFunction(&candidate.function, candidate.module, symbol), "candidate symbol");
    base.grid_x = 24;
    base.block_x = 256;
    base.lds = 49408;
    candidate.grid_x = 8;
    candidate.block_x = 768;
    candidate.lds = 37120;

    Rng rng;
    std::vector<float> q_host(static_cast<size_t>(BATCH) * NH * HD);
    for (float& x : q_host)
        x = (static_cast<float>(rng.next() % 20001) / 10000.0f - 1.0f) * 4.0f;
    std::vector<uint8_t> k_host(static_cast<size_t>(MAX_CTX) * ROW_BYTES);
    std::vector<uint8_t> v_host(static_cast<size_t>(MAX_CTX) * ROW_BYTES);
    for (int row = 0; row < MAX_CTX; ++row) {
        const size_t base_off = static_cast<size_t>(row) * ROW_BYTES;
        for (int d = 0; d < 1024; ++d) {
            uint8_t kc = static_cast<uint8_t>(rng.next());
            uint8_t vc = static_cast<uint8_t>(rng.next());
            if ((kc & 0x7f) == 0x7f) kc ^= 1;
            if ((vc & 0x7f) == 0x7f) vc ^= 1;
            k_host[base_off + d] = kc;
            v_host[base_off + d] = vc;
        }
        for (int kh = 0; kh < NKV; ++kh) {
            const uint16_t ks = 0x3000u + ((row * 37 + kh * 211) & 0x0fff);
            const uint16_t vs = 0x2c00u + ((row * 53 + kh * 173) & 0x0fff);
            std::memcpy(k_host.data() + base_off + 1024 + 2 * kh, &ks, 2);
            std::memcpy(v_host.data() + base_off + 1024 + 2 * kh, &vs, 2);
        }
    }

    void *q = nullptr, *q8 = nullptr, *k = nullptr, *v = nullptr;
    void *out = nullptr, *positions = nullptr;
    check(hipMalloc(&q, q_host.size() * sizeof(float)), "hipMalloc Q");
    check(hipMalloc(&q8, static_cast<size_t>(BATCH) * NH * HD), "hipMalloc Q8");
    check(hipMalloc(&k, k_host.size()), "hipMalloc K");
    check(hipMalloc(&v, v_host.size()), "hipMalloc V");
    check(hipMalloc(&out, static_cast<size_t>(BATCH) * NH * HD * sizeof(float)), "hipMalloc out");
    check(hipMalloc(&positions, static_cast<size_t>(BATCH) * sizeof(int)), "hipMalloc positions");
    check(hipMemcpy(q, q_host.data(), q_host.size() * sizeof(float), hipMemcpyHostToDevice), "copy Q");
    check(hipMemcpy(k, k_host.data(), k_host.size(), hipMemcpyHostToDevice), "copy K");
    check(hipMemcpy(v, v_host.data(), v_host.size(), hipMemcpyHostToDevice), "copy V");

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\"schema\":\"attn-lane-pinned-body-v1\",\"batch\":" << BATCH
              << ",\"warmups\":" << WARMUPS << ",\"samples\":" << SAMPLES << "}\n";
    for (int ctx : LENGTHS) {
        std::vector<int> pos(BATCH);
        const int first = std::max(0, ctx - BATCH);
        for (int i = 0; i < BATCH; ++i) pos[i] = std::min(ctx - 1, first + i);
        check(hipMemcpy(positions, pos.data(), pos.size() * sizeof(int), hipMemcpyHostToDevice),
              "copy positions");
        for (int i = 0; i < WARMUPS; ++i) {
            sample(base, q, q8, k, v, out, positions);
            sample(candidate, q, q8, k, v, out, positions);
        }
        std::vector<double> a1, b, a2;
        for (int i = 0; i < SAMPLES; ++i) {
            if ((i & 1) == 0) {
                a1.push_back(sample(base, q, q8, k, v, out, positions));
                b.push_back(sample(candidate, q, q8, k, v, out, positions));
                a2.push_back(sample(base, q, q8, k, v, out, positions));
            } else {
                a2.push_back(sample(base, q, q8, k, v, out, positions));
                b.push_back(sample(candidate, q, q8, k, v, out, positions));
                a1.push_back(sample(base, q, q8, k, v, out, positions));
            }
        }
        const Stats s1 = stats(std::move(a1));
        const Stats sb = stats(std::move(b));
        const Stats s2 = stats(std::move(a2));
        const double baseline = 0.5 * (s1.median + s2.median);
        std::cout << "{\"ctx\":" << ctx
                  << ",\"a1_median_us\":" << s1.median
                  << ",\"a1_p95_us\":" << s1.p95
                  << ",\"candidate_median_us\":" << sb.median
                  << ",\"candidate_p95_us\":" << sb.p95
                  << ",\"a2_median_us\":" << s2.median
                  << ",\"a2_p95_us\":" << s2.p95
                  << ",\"aa_drift_pct\":" << (s2.median / s1.median - 1.0) * 100.0
                  << ",\"candidate_delta_pct\":" << (baseline / sb.median - 1.0) * 100.0
                  << ",\"a1_sorted_us\":";
        print_samples(s1.sorted);
        std::cout << ",\"candidate_sorted_us\":";
        print_samples(sb.sorted);
        std::cout << ",\"a2_sorted_us\":";
        print_samples(s2.sorted);
        std::cout << "}\n";
    }

    check(hipFree(q), "hipFree Q");
    check(hipFree(q8), "hipFree Q8");
    check(hipFree(k), "hipFree K");
    check(hipFree(v), "hipFree V");
    check(hipFree(out), "hipFree out");
    check(hipFree(positions), "hipFree positions");
    check(hipModuleUnload(base.module), "unload baseline");
    check(hipModuleUnload(candidate.module), "unload candidate");
    return 0;
}
