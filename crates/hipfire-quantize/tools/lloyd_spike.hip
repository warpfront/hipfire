// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// P1 GPU oracle spike for the per-tensor Lloyd fit. Reads the binary fixture
// written by `quant_mq4v2_lloyd::tests::lloyd_gpu_fixture` and checks that a
// GPU chunk reduction is BITWISE equal to the CPU chunk accumulation that is
// itself byte-identical to the serial production path on the 0.8B fixture.
//
// Build:  hipcc -O3 -ffp-contract=off --offload-arch=gfx1100 \
//             lloyd_spike.hip -o lloyd_spike.exe
// Run:    lloyd_spike.exe <fixture.bin>
//
// Layout (all little-endian): u64 n; f64[16] levels; f64[n] vals; f64[n] wts;
// f64[16] sw; f64[16] sv; f64 sse; f64 wsum.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>

static constexpr int CHUNK = 1024;
static constexpr int LANES = 34;  // sw[16] sv[16] sse wsum

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

static bool read_u64(FILE* f, uint64_t& out) { return fread(&out, 8, 1, f) == 1; }
static bool read_f64(FILE* f, double& out) { return fread(&out, 8, 1, f) == 1; }

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: %s <fixture.bin>\n", argv[0]); return 2; }
  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::perror("open"); return 2; }
  uint64_t n = 0;
  if (!read_u64(f, n)) { std::fprintf(stderr, "short header\n"); return 2; }
  std::vector<double> levels(16), vals(n), wts(n), sw(16), sv(16);
  for (auto& v : levels) if (!read_f64(f, v)) return 2;
  for (auto& v : vals) if (!read_f64(f, v)) return 2;
  for (auto& v : wts) if (!read_f64(f, v)) return 2;
  for (auto& v : sw) if (!read_f64(f, v)) return 2;
  for (auto& v : sv) if (!read_f64(f, v)) return 2;
  double sse = 0.0, wsum = 0.0;
  if (!read_f64(f, sse) || !read_f64(f, wsum)) return 2;
  std::fclose(f);

  hipInit(0);
  int dev = 0;
  hipGetDevice(&dev);
  hipDeviceProp_t prop{};
  hipGetDeviceProperties(&prop, dev);
  std::printf("device: %s (arch gfx%d%d)\n", prop.name, prop.major, prop.minor);

  const size_t nbytes = static_cast<size_t>(n) * sizeof(double);
  double *d_vals = nullptr, *d_wts = nullptr, *d_levels = nullptr, *d_partials = nullptr;
  hipMalloc(&d_vals, nbytes);
  hipMalloc(&d_wts, nbytes);
  hipMalloc(&d_levels, 16 * sizeof(double));
  const uint64_t nchunks = (n + CHUNK - 1) / CHUNK;
  hipMalloc(&d_partials, static_cast<size_t>(nchunks) * LANES * sizeof(double));
  auto t_h2d0 = std::chrono::steady_clock::now();
  hipMemcpy(d_vals, vals.data(), nbytes, hipMemcpyHostToDevice);
  hipMemcpy(d_wts, wts.data(), nbytes, hipMemcpyHostToDevice);
  hipMemcpy(d_levels, levels.data(), 16 * sizeof(double), hipMemcpyHostToDevice);
  hipDeviceSynchronize();
  double h2d_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t_h2d0).count();

  const int block = 256;
  const unsigned grid = static_cast<unsigned>((nchunks + block - 1) / block);
  hipLaunchKernelGGL(lloyd_partial, dim3(grid), dim3(block), 0, 0,
                     d_vals, d_wts, d_levels, d_partials, n);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) { std::fprintf(stderr, "launch: %s\n", hipGetErrorString(err)); return 1; }

  std::vector<double> partials(static_cast<size_t>(nchunks) * LANES);
  auto t_d2h0 = std::chrono::steady_clock::now();
  hipMemcpy(partials.data(), d_partials, partials.size() * sizeof(double), hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  double d2h_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t_d2h0).count();

  // Repeated launches with the pool resident on device: the steady-state
  // cost of one Lloyd iteration (levels change every real iteration; here
  // they are fixed, so this measures the kernel, not the full fit).
  std::vector<double> times;
  for (int rep = 0; rep < 30; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(lloyd_partial, dim3(grid), dim3(block), 0, 0,
                       d_vals, d_wts, d_levels, d_partials, n);
    hipDeviceSynchronize();
    times.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(times.begin(), times.end());

  // Combine chunk partials in chunk order (the CPU tree-experiment order).
  double g_sw[16] = {0.0}, g_sv[16] = {0.0}, g_sse = 0.0, g_wsum = 0.0;
  for (uint64_t c = 0; c < nchunks; ++c) {
    const double* p = partials.data() + static_cast<size_t>(c) * LANES;
    for (int k = 0; k < 16; ++k) { g_sw[k] += p[k]; g_sv[k] += p[16 + k]; }
    g_sse += p[32];
    g_wsum += p[33];
  }

  int bad = 0;
  auto check = [&](const char* name, double got, double want) {
    uint64_t a = 0, b = 0;
    std::memcpy(&a, &got, 8);
    std::memcpy(&b, &want, 8);
    if (a != b) {
      if (bad < 8) std::printf("  MISMATCH %s: got %.17g want %.17g (bits %016llx vs %016llx)\n",
                               name, got, want, (unsigned long long)a, (unsigned long long)b);
      ++bad;
    }
  };
  for (int k = 0; k < 16; ++k) {
    char name[32];
    std::snprintf(name, sizeof(name), "sw[%d]", k); check(name, g_sw[k], sw[k]);
    std::snprintf(name, sizeof(name), "sv[%d]", k); check(name, g_sv[k], sv[k]);
  }
  check("sse", g_sse, sse);
  check("wsum", g_wsum, wsum);

  std::printf("n=%llu chunks=%llu grid=%u block=%d : %s (%d/34 words differ)\n",
              (unsigned long long)n, (unsigned long long)nchunks, grid, block,
              bad == 0 ? "BITWISE PASS" : "FAIL", bad);
  std::printf("timing ms: h2d=%.3f d2h=%.3f kernel median=%.3f min=%.3f max=%.3f\n",
              h2d_ms, d2h_ms, times[times.size() / 2], times.front(), times.back());

  hipFree(d_vals); hipFree(d_wts); hipFree(d_levels); hipFree(d_partials);
  return bad == 0 ? 0 : 1;
}
