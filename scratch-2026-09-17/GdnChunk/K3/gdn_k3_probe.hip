// Standalone GDN K3 gate probe. Intentionally kept out of git.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../../kernels/src/gdn_recompute_w_u.gfx1201.hip"

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t e_ = (call);                                                \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,      \
                         #call, hipGetErrorString(e_));                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static double half_rounded(float x) { return (double)(_Float16)x; }

int main() {
    constexpr int N = 512;
    constexpr int H = 48;
    constexpr int D = 128;
    constexpr int C = 64;
    constexpr int J = (N + C - 1) / C;
    constexpr int warmups = 5;
    constexpr int samples = 20;

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int active_blocks_per_mp = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_mp, gdn_recompute_w_u_gfx1201, 256, 0));

    const size_t x_count = (size_t)N * H * D;
    const size_t beta_count = (size_t)N * H;
    const size_t b_count = (size_t)J * H * C * C;
    const size_t g_count = (size_t)J * H * C;
    const size_t out_count = (size_t)J * H * C * D;
    std::vector<float> hk(x_count), hv(x_count), hbeta(beta_count);
    std::vector<float> hB(b_count), hG(g_count);

    for (int t = 0; t < N; ++t) {
        for (int h = 0; h < H; ++h) {
            hbeta[(size_t)t * H + h] =
                0.08f + 0.84f * (float)((t * 17 + h * 11) % 101) / 100.0f;
            for (int d = 0; d < D; ++d) {
                const size_t x = ((size_t)t * H + h) * D + d;
                hk[x] = 0.078f * (std::sin(0.013f * (float)(1 + t * 131 + h * 17 + d)) +
                                  0.37f * std::cos(0.021f * (float)(3 + t * 19 + h * 127 + d * 5)));
                hv[x] = 0.31f * std::sin(0.007f * (float)(5 + t * 29 + h * 7 + d * 3));
            }
        }
    }
    for (int c = 0; c < J; ++c) {
        for (int h = 0; h < H; ++h) {
            float run = 0.0f;
            const size_t gb = ((size_t)c * H + h) * C;
            const size_t bb = ((size_t)c * H + h) * C * C;
            for (int r = 0; r < C; ++r) {
                run += -0.0015f - 0.00011f * (float)(((c * C + r) * 7 + h * 13) % 47);
                hG[gb + r] = run;
                for (int col = 0; col < C; ++col) {
                    float value = 0.0f;
                    if (r == col) value = 1.0f;
                    else if (col < r)
                        value = 0.003f * std::sin(0.11f * (float)(1 + r * 67 + col * 13 + h));
                    hB[bb + (size_t)r * C + col] = value;
                }
            }
        }
    }

    float *dk = nullptr, *dv = nullptr, *dbeta = nullptr, *dB = nullptr, *dG = nullptr;
    _Float16 *dw = nullptr, *du = nullptr;
    HIP_CHECK(hipMalloc(&dk, x_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&dv, x_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&dbeta, beta_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, b_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&dG, g_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&dw, out_count * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&du, out_count * sizeof(_Float16)));
    HIP_CHECK(hipMemcpy(dk, hk.data(), x_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dv, hv.data(), x_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dbeta, hbeta.data(), beta_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), b_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dG, hG.data(), g_count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dw, 0xA5, out_count * sizeof(_Float16)));
    HIP_CHECK(hipMemset(du, 0xA5, out_count * sizeof(_Float16)));

    const dim3 grid(J, H, 1), block(256, 1, 1);
    for (int i = 0; i < warmups; ++i) {
        hipLaunchKernelGGL(gdn_recompute_w_u_gfx1201, grid, block, 0, 0,
                           dk, dv, dbeta, dB, dG, dw, du, N, H);
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
        hipLaunchKernelGGL(gdn_recompute_w_u_gfx1201, grid, block, 0, 0,
                           dk, dv, dbeta, dB, dG, dw, du, N, H);
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

    std::vector<_Float16> got_w(C * D), got_u(C * D);
    HIP_CHECK(hipMemcpy(got_w.data(), dw, got_w.size() * sizeof(_Float16),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got_u.data(), du, got_u.size() * sizeof(_Float16),
                        hipMemcpyDeviceToHost));
    double w_max_abs = 0.0, u_max_abs = 0.0;
    for (int r = 0; r < C; ++r) {
        for (int col = 0; col < D; ++col) {
            double w_ref = 0.0, u_ref = 0.0;
            for (int kk = 0; kk < C; ++kk) {
                const double bv = half_rounded(hB[(size_t)r * C + kk]);
                const double beta_v = (double)hbeta[(size_t)kk * H];
                const double krhs = half_rounded((float)(beta_v * std::exp((double)hG[kk]) *
                                                        (double)hk[(size_t)kk * H * D + col]));
                const double vrhs = half_rounded((float)(beta_v *
                                                        (double)hv[(size_t)kk * H * D + col]));
                w_ref += bv * krhs;
                u_ref += bv * vrhs;
            }
            const double w_err = std::abs((double)got_w[(size_t)r * D + col] - w_ref);
            const double u_err = std::abs((double)got_u[(size_t)r * D + col] - u_ref);
            w_max_abs = std::max(w_max_abs, w_err);
            u_max_abs = std::max(u_max_abs, u_err);
        }
    }

    // Tail contract: invalid rows in the final C64 are explicitly zero.
    constexpr int tail_n = 493;
    HIP_CHECK(hipMemset(dw, 0xA5, out_count * sizeof(_Float16)));
    HIP_CHECK(hipMemset(du, 0xA5, out_count * sizeof(_Float16)));
    hipLaunchKernelGGL(gdn_recompute_w_u_gfx1201, grid, block, 0, 0,
                       dk, dv, dbeta, dB, dG, dw, du, tail_n, H);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const size_t tail_base = ((size_t)(J - 1) * H) * C * D;
    std::vector<_Float16> tail(C * D);
    HIP_CHECK(hipMemcpy(tail.data(), dw + tail_base, tail.size() * sizeof(_Float16),
                        hipMemcpyDeviceToHost));
    double invalid_tail_max = 0.0;
    for (int r = tail_n - (J - 1) * C; r < C; ++r)
        for (int col = 0; col < D; ++col)
            invalid_tail_max = std::max(invalid_tail_max,
                                        std::abs((double)tail[(size_t)r * D + col]));

    std::printf("device=%s\n", prop.name);
    std::printf("geometry=N%d H%d D%d C%d grid=(%d,%d,1) block=(256,1,1) rows=%d\n",
                N, H, D, C, J, H, N);
    std::printf("runtime_active_blocks_per_mp=%d\n", active_blocks_per_mp);
    std::printf("warmups=%d samples=%d\n", warmups, samples);
    std::printf("raw_event_us=");
    for (int i = 0; i < samples; ++i)
        std::printf("%s%.3f", i ? "," : "", raw_us[i]);
    std::printf("\nmedian_event_us=%.3f\n", median_us);
    std::printf("cpu_f64_first_c64_w_max_abs_delta=%.9g\n", w_max_abs);
    std::printf("cpu_f64_first_c64_u_max_abs_delta=%.9g\n", u_max_abs);
    std::printf("tail_N%d_invalid_row_max_abs=%.9g\n", tail_n, invalid_tail_max);
    std::printf("gate=%s\n",
                (active_blocks_per_mp >= 2 && std::isfinite(w_max_abs) &&
                 std::isfinite(u_max_abs) && w_max_abs < 0.01 && u_max_abs < 0.01 &&
                 invalid_tail_max == 0.0) ? "ALIVE" : "KILL");

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(dk)); HIP_CHECK(hipFree(dv)); HIP_CHECK(hipFree(dbeta));
    HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dG)); HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(du));
    return (active_blocks_per_mp >= 2 && std::isfinite(w_max_abs) &&
            std::isfinite(u_max_abs) && w_max_abs < 0.01 && u_max_abs < 0.01 &&
            invalid_tail_max == 0.0) ? 0 : 2;
}
