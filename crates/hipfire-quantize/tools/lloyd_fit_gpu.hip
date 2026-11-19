// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Stream (2) research: full 25-iteration unconstrained Lloyd fit on the GPU.
// Reads the fixture written by `quant_mq4v2_lloyd::tests::lloyd_fit_fixture`
// and checks that the final codebook is BITWISE equal to the CPU fit.
//
// Fixture layout (little-endian): u64 n; f64[16] expected levels;
// f64[n] vals; f64[n] wts.
//
// Build: hipcc -O3 -ffp-contract=off --offload-arch=gfx1100 \
//             -D_CRT_SECURE_NO_WARNINGS lloyd_fit_gpu.hip -o lloyd_fit_gpu.exe
// Run:   lloyd_fit_gpu.exe <fixture.bin>

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int CHUNK = 1024;
static constexpr int LANES = 34;  // sw[16] sv[16] sse wsum
static constexpr int MAX_ITERS = 25;

__global__ void lloyd_partial(const double* __restrict__ vals,
                              const double* __restrict__ wts,
                              const double* __restrict__ levels,
                              double* __restrict__ partials,
                              uint64_t n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t start = static_cast<uint64_t>(tid) * CHUNK;
  if (start >= n) return;
  double L[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) L[k] = levels[k];
  double sw[16] = {0.0}, sv[16] = {0.0};
  double sse = 0.0, wsum = 0.0;
  for (uint64_t i = start; i < start + CHUNK && i < n; ++i) {
    const double v = vals[i], wt = wts[i];
    int idx = 15;
#pragma unroll
    for (int k = 0; k < 15; ++k) {
      const double mid = 0.5 * (L[k] + L[k + 1]);
      if (v < mid) { idx = k; break; }
    }
    sw[idx] += wt;
    sv[idx] += wt * v;
    const double e = v - L[idx];
    sse += wt * e * e;
    wsum += wt;
  }
  double* p = partials + static_cast<size_t>(tid) * LANES;
#pragma unroll
  for (int k = 0; k < 16; ++k) { p[k] = sw[k]; p[16 + k] = sv[k]; }
  p[32] = sse;
  p[33] = wsum;
}

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: %s <fixture.bin>\n", argv[0]); return 2; }
  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::perror("open"); return 2; }
  uint64_t n = 0;
  if (fread(&n, 8, 1, f) != 1) return 2;
  std::vector<double> expected(16), vals(n), wts(n);
  for (auto& v : expected) if (fread(&v, 8, 1, f) != 1) return 2;
  for (auto& v : vals) if (fread(&v, 8, 1, f) != 1) return 2;
  for (auto& v : wts) if (fread(&v, 8, 1, f) != 1) return 2;
  std::fclose(f);

  hipInit(0);
  int dev = 0;
  hipGetDevice(&dev);
  hipDeviceProp_t prop{};
  hipGetDeviceProperties(&prop, dev);
  std::printf("device: %s\n", prop.name);

  const size_t nbytes = static_cast<size_t>(n) * sizeof(double);
  double *d_vals = nullptr, *d_wts = nullptr, *d_levels = nullptr, *d_partials = nullptr;
  hipMalloc(&d_vals, nbytes);
  hipMalloc(&d_wts, nbytes);
  hipMalloc(&d_levels, 16 * sizeof(double));
  const uint64_t nchunks = (n + CHUNK - 1) / CHUNK;
  hipMalloc(&d_partials, static_cast<size_t>(nchunks) * LANES * sizeof(double));
  hipMemcpy(d_vals, vals.data(), nbytes, hipMemcpyHostToDevice);
  hipMemcpy(d_wts, wts.data(), nbytes, hipMemcpyHostToDevice);
  hipDeviceSynchronize();

  const int block = 256;
  const unsigned grid = static_cast<unsigned>((nchunks + block - 1) / block);
  std::vector<double> partials(static_cast<size_t>(nchunks) * LANES);
  std::vector<double> levels(16);
  for (int k = 0; k < 16; ++k) levels[k] = static_cast<double>(k);

  int iters = 0;
  double total_kernel_ms = 0.0, total_d2h_ms = 0.0;
  for (int it = 0; it < MAX_ITERS; ++it) {
    hipMemcpy(d_levels, levels.data(), 16 * sizeof(double), hipMemcpyHostToDevice);

    auto k0 = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(lloyd_partial, dim3(grid), dim3(block), 0, 0,
                       d_vals, d_wts, d_levels, d_partials, n);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) { std::fprintf(stderr, "launch: %s\n", hipGetErrorString(err)); return 1; }
    hipDeviceSynchronize();
    total_kernel_ms += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - k0).count();

    auto d0 = std::chrono::steady_clock::now();
    hipMemcpy(partials.data(), d_partials, partials.size() * sizeof(double), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    total_d2h_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - d0).count();

    double sw[16] = {0.0}, sv[16] = {0.0}, sse = 0.0, wsum = 0.0;
    for (uint64_t c = 0; c < nchunks; ++c) {
      const double* p = partials.data() + static_cast<size_t>(c) * LANES;
      for (int k = 0; k < 16; ++k) { sw[k] += p[k]; sv[k] += p[16 + k]; }
      sse += p[32];
      wsum += p[33];
    }
    const double mse = wsum > 0.0 ? sse / wsum : 0.0;

    double next[16];
    for (int k = 0; k < 16; ++k) next[k] = sw[k] > 0.0 ? sv[k] / sw[k] : levels[k];
    std::stable_sort(next, next + 16);
    double moved = 0.0;
    for (int k = 0; k < 16; ++k) moved = std::max(moved, std::fabs(next[k] - levels[k]));
    std::copy(next, next + 16, levels.begin());
    ++iters;
    if (it == 0) std::printf("iter 0 mse=%.17g\n", mse);
    if (moved < 1e-6) break;
  }

  int bad = 0;
  for (int k = 0; k < 16; ++k) {
    uint64_t a = 0, b = 0;
    std::memcpy(&a, &levels[k], 8);
    std::memcpy(&b, &expected[k], 8);
    if (a != b) {
      if (bad < 6) std::printf("  MISMATCH [%d]: gpu %.17g cpu %.17g\n", k, levels[k], expected[k]);
      ++bad;
    }
  }
  std::printf("n=%llu iters=%d : %s (%d/16 levels differ)\n",
              (unsigned long long)n, iters, bad == 0 ? "BITWISE PASS" : "FAIL", bad);
  std::printf("timing ms: kernel total=%.3f (%.3f/iter) d2h total=%.3f (%.3f/iter)\n",
              total_kernel_ms, total_kernel_ms / iters,
              total_d2h_ms, total_d2h_ms / iters);

  hipFree(d_vals); hipFree(d_wts); hipFree(d_levels); hipFree(d_partials);
  return bad == 0 ? 0 : 1;
}
