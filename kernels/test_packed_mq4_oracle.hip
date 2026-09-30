// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 HUSRCF
// Standalone layout oracle, not an end-to-end quality or performance gate.
// hipcc --offload-arch=gfx1100 -O3 kernels/test_packed_mq4_oracle.hip -o /tmp/packed-oracle
// Repeat with -DPACKED_MQ4_V2=1 for the uniform MQ4V2 header.
#include "src/gemm_mq4_packed.gfx1100.hip"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(call) do { auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); std::exit(2); \
} } while (0)

int main(int argc, char** argv) {
    constexpr int M = 128, N = 512;
    const int K = argc > 1 ? std::atoi(argv[1]) : 512;
    if (K <= 0 || K % 256 != 0) {
        std::fprintf(stderr, "K must be positive and divisible by 256\n");
        return 2;
    }
    std::vector<unsigned char> weights(M * (K / 256) * 136);
    std::vector<float> x(N * K), y(N * M), reference(N * M);
    for (int row = 0; row < M; ++row) {
        for (int group = 0; group < K / 256; ++group) {
            auto* p = weights.data() + (row * (K / 256) + group) * 136;
#ifdef PACKED_MQ4_V2
            const half header[4] = {half(0.125f * (1 + row % 3)), half(-0.5f),
                                    half(0.25f * (1 + group % 3)), half(0.75f)};
#else
            const float header[2] = {0.125f * (1 + row % 3), -0.5f};
#endif
            std::memcpy(p, header, 8);
            for (int i = 0; i < 128; ++i) {
                p[8 + i] = ((14 * i + i / 8 + 3 * row + 5 * group) % 16)
                    | (((14 * i + 7 + i / 8 + 3 * row + 5 * group) % 16) << 4);
            }
        }
    }
    // Exactly representable Q8 quantization, but distinct per-token/per-half
    // scales and irregular mixed signs. Per-32 sums also remain half-exact.
    for (int token = 0; token < N; ++token) {
        for (int i = 0; i < K; ++i) {
            const float scale = std::ldexp(1.0f, (token + i / 128) % 5 - 2);
            const int q = i % 128 == 0 ? 127
                : ((i * 17 + token * 29 + i / 128 * 7) % 15 - 7);
            x[token * K + i] = q * scale;
        }
    }
    for (int token = 0; token < N; ++token) {
        for (int row = 0; row < M; ++row) {
            double sum = 0;
            for (int i = 0; i < K; ++i) {
                const auto* p = weights.data() + (row * (K / 256) + i / 256) * 136;
                float scale, zero;
#ifdef PACKED_MQ4_V2
                half header[4]; std::memcpy(header, p, 8);
                const int h = (i % 256) / 128;
                scale = float(header[2 * h]); zero = float(header[2 * h + 1]);
#else
                float header[2]; std::memcpy(header, p, 8);
                scale = header[0]; zero = header[1];
#endif
                const int q = (p[8 + (i % 256) / 2] >> (4 * (i % 2))) & 15;
                sum += (scale * q + zero) * x[token * K + i];
            }
            reference[token * M + row] = float(sum);
        }
    }
    unsigned char* dw; float *dx, *dy; block_q8_1_mmq* dq;
    HIP_OK(hipMalloc(&dw, weights.size()));
    HIP_OK(hipMalloc(&dx, x.size() * sizeof(float)));
    HIP_OK(hipMalloc(&dy, y.size() * sizeof(float)));
    HIP_OK(hipMalloc(&dq, (K / 128) * N * sizeof(block_q8_1_mmq)));
    HIP_OK(hipMemcpy(dw, weights.data(), weights.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(quantize_mq4_packed_group128, dim3((K + 1023) / 1024, N),
                      dim3(256), 0, 0, dx, dq, K, N);
    HIP_OK(hipGetLastError());
    for (bool add : {false, true}) {
        std::fill(y.begin(), y.end(), 7.0f);
        HIP_OK(hipMemcpy(dy, y.data(), y.size() * sizeof(float), hipMemcpyHostToDevice));
        if (add) {
            hipLaunchKernelGGL(gemm_mq4_packed_gfx1100_full_add, dim3(M / 64, N / 256),
                              dim3(32, 8), 57344, 0, (const char*)dw, dq, dy, M, K, N, 1);
        } else {
            hipLaunchKernelGGL(gemm_mq4_packed_gfx1100_full_set, dim3(M / 64, N / 256),
                              dim3(32, 8), 57344, 0, (const char*)dw, dq, dy, M, K, N, 0);
        }
        HIP_OK(hipGetLastError());
        HIP_OK(hipDeviceSynchronize());
        HIP_OK(hipMemcpy(y.data(), dy, y.size() * sizeof(float), hipMemcpyDeviceToHost));
        float max_error = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            float error = std::fabs(y[i] - reference[i] - (add ? 7.0f : 0.0f));
            if (!std::isfinite(y[i]) || error > 0.01f) {
                std::fprintf(stderr, "FAIL add=%d index=%zu actual=%g expected=%g\n",
                             add, i, y[i], reference[i] + (add ? 7.0f : 0.0f));
                return 1;
            }
            max_error = std::fmax(max_error, error);
        }
        std::printf("PASS K=%d add=%d values=%zu max_abs_error=%g\n", K, add, y.size(), max_error);
    }
    HIP_OK(hipFree(dq)); HIP_OK(hipFree(dy)); HIP_OK(hipFree(dx)); HIP_OK(hipFree(dw));
}
