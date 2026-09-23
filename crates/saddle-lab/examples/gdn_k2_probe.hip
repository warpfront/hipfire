// Standalone GDN K2 kill-gate probe. Intentionally kept out of git.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../../kernels/src/gdn_kkt_bc16_solve.gfx1201.hip"

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t e_ = (call);                                                \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,      \
                         #call, hipGetErrorString(e_));                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static double half_rounded(float x) {
    return (double)(_Float16)x;
}

int main() {
    constexpr int N = 512;
    constexpr int H = 48;
    constexpr int D = 128;
    constexpr int C = 64;
    constexpr int BC = 16;
    constexpr int J = (N + C - 1) / C;
    constexpr int warmups = 5;
    constexpr int samples = 20;

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int active_blocks_per_mp = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_mp, gdn_kkt_bc16_solve_gfx1201, 256, 0));

    const size_t k_count = (size_t)N * H * D;
    const size_t gh_count = (size_t)N * H;
    const size_t b_count = (size_t)J * H * C * C;
    const size_t g_count = (size_t)J * H * C;
    std::vector<float> hk(k_count), hgate(gh_count), hbeta(gh_count);

    // Deterministic nonzero fixture: roughly unit-norm keys, modest negative
    // log-decay, and beta spanning the prepared sigmoid's useful interior.
    for (int t = 0; t < N; ++t) {
        for (int h = 0; h < H; ++h) {
            hgate[(size_t)t * H + h] =
                -0.0015f - 0.00011f * (float)((t * 7 + h * 13) % 47);
            hbeta[(size_t)t * H + h] =
                0.08f + 0.84f * (float)((t * 17 + h * 11) % 101) / 100.0f;
            for (int d = 0; d < D; ++d) {
                const float a = std::sin(0.013f * (float)(1 + t * 131 + h * 17 + d));
                const float b = std::cos(0.021f * (float)(3 + t * 19 + h * 127 + d * 5));
                hk[((size_t)t * H + h) * D + d] = (a + 0.37f * b) * 0.078f;
            }
        }
    }

    float *dk = nullptr, *dgate = nullptr, *dbeta = nullptr;
    float *dB = nullptr, *dG = nullptr;
    HIP_CHECK(hipMalloc(&dk, k_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&dgate, gh_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&dbeta, gh_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, b_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&dG, g_count * sizeof(float)));
    HIP_CHECK(hipMemcpy(dk, hk.data(), k_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dgate, hgate.data(), gh_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbeta, hbeta.data(), gh_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dB, 0xA5, b_count * sizeof(float)));
    HIP_CHECK(hipMemset(dG, 0xA5, g_count * sizeof(float)));

    const dim3 grid(J, H, 1);
    const dim3 block(256, 1, 1);
    for (int i = 0; i < warmups; ++i) {
        hipLaunchKernelGGL(gdn_kkt_bc16_solve_gfx1201, grid, block, 0, 0,
                           dk, dgate, dbeta, dB, dG, N, H);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> raw_us;
    raw_us.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        HIP_CHECK(hipEventRecord(start, nullptr));
        hipLaunchKernelGGL(gdn_kkt_bc16_solve_gfx1201, grid, block, 0, 0,
                           dk, dgate, dbeta, dB, dG, N, H);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(stop, nullptr));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        raw_us.push_back(ms * 1000.0f);
    }

    std::vector<float> sorted = raw_us;
    std::sort(sorted.begin(), sorted.end());
    const float median_us = 0.5f * (sorted[9] + sorted[10]);

    // CPU f64 oracle for the complete first C64 inverse. K operands are first
    // rounded to f16, matching the kernel's WMMA input construction.
    std::vector<double> prefix(C), L(C * C, 0.0), inv(C * C, 0.0);
    float prefix_f32 = 0.0f;
    for (int i = 0; i < C; ++i) {
        prefix_f32 += hgate[(size_t)i * H];
        prefix[i] = (double)prefix_f32;
    }
    for (int i = 0; i < C; ++i) {
        for (int j = 0; j < i; ++j) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d) {
                dot += half_rounded(hk[(size_t)i * H * D + d]) *
                       half_rounded(hk[(size_t)j * H * D + d]);
            }
            L[i * C + j] = (double)hbeta[(size_t)i * H] *
                            std::exp(prefix[i] - prefix[j]) * dot;
        }
    }
    for (int i = 0; i < C; ++i) {
        for (int col = 0; col < C; ++col) {
            double value = (i == col) ? 1.0 : 0.0;
            for (int r = 0; r < i; ++r)
                value -= L[i * C + r] * inv[r * C + col];
            inv[i * C + col] = value;
        }
    }

    std::vector<float> got(C * C);
    HIP_CHECK(hipMemcpy(got.data(), dB, got.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    double block_max_abs = 0.0;
    double chunk_max_abs = 0.0;
    for (int i = 0; i < C; ++i) {
        for (int j = 0; j < C; ++j) {
            const double e = std::abs((double)got[i * C + j] - inv[i * C + j]);
            chunk_max_abs = std::max(chunk_max_abs, e);
            if (i < BC && j < BC) block_max_abs = std::max(block_max_abs, e);
        }
    }
    double residual_inf = 0.0;
    for (int i = 0; i < C; ++i) {
        double rowsum = 0.0;
        for (int j = 0; j < C; ++j) {
            double v = (double)got[i * C + j] -
                       ((i == j) ? 1.0 : 0.0);
            for (int r = 0; r < i; ++r)
                v += L[i * C + r] * (double)got[r * C + j];
            rowsum += std::abs(v);
        }
        residual_inf = std::max(residual_inf, rowsum);
    }

    std::printf("device=%s\n", prop.name);
    std::printf("geometry=N%d H%d dk%d C%d BC%d grid=(%d,%d,1) block=(256,1,1) rows=%d\n",
                N, H, D, C, BC, J, H, N);
    std::printf("runtime_active_blocks_per_mp=%d\n", active_blocks_per_mp);
    std::printf("warmups=%d samples=%d\n", warmups, samples);
    std::printf("raw_event_us=");
    for (int i = 0; i < samples; ++i)
        std::printf("%s%.3f", i ? "," : "", raw_us[i]);
    std::printf("\nmedian_event_us=%.3f\n", median_us);
    std::printf("cpu_f64_first_bc16_max_abs_delta=%.9g\n", block_max_abs);
    std::printf("cpu_f64_first_c64_max_abs_delta=%.9g\n", chunk_max_abs);
    std::printf("first_c64_inverse_residual_inf=%.9g\n", residual_inf);
    std::printf("projected_pp8192_us_per_token=%.6f\n",
                (double)median_us * (8192.0 / 512.0) * 48.0 / 8192.0);
    std::printf("gate=%s\n", median_us < 200.0f ? "ALIVE" : "KILL");

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(dk));
    HIP_CHECK(hipFree(dgate));
    HIP_CHECK(hipFree(dbeta));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dG));
    return (std::isfinite(chunk_max_abs) && chunk_max_abs < 0.01) ? 0 : 2;
}
