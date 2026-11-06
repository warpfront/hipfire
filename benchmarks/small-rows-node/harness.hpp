// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Standalone node harness: BF16-weight small-M GEMMs (router 128x2560, selector 17x10240).
#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)

struct Lcg { uint64_t s; float next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return ((s >> 40) / float(1u << 24)) * 2.f - 1.f; } };
static inline uint16_t f2bf(float x) { uint32_t u; memcpy(&u, &x, 4); if ((u & 0x7fffffff) > 0x7f800000) return u >> 16; u += 0x7fff + ((u >> 16) & 1); return u >> 16; }
static inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

struct Geo { const char* name; int M, K; bool xbf16; };
// launch(W, X, Y, M, K, N, stream)
using Launch = std::function<void(const uint16_t*, const void*, float*, int, int, int)>;
struct Arm { std::string name; Launch fn; int minN, maxN; int onlyM = 0; };

inline double time_arm(const Arm& a, const uint16_t* W, void** xs, int nx, float* Y, int M, int K, int N, int& iters_out) {
  hipEvent_t s, e; CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
  for (int i = 0; i < 3; ++i) a.fn(W, xs[i % nx], Y, M, K, N);
  CK(hipDeviceSynchronize());
  // calibrate
  int iters = 8;
  float ms = 0;
  for (;;) {
    CK(hipEventRecord(s));
    for (int i = 0; i < iters; ++i) a.fn(W, xs[i % nx], Y, M, K, N);
    CK(hipEventRecord(e)); CK(hipEventSynchronize(e));
    CK(hipEventElapsedTime(&ms, s, e));
    if (ms >= 40.f || iters >= 20000) break;
    iters *= 2;
  }
  std::vector<double> samp;
  for (int r = 0; r < 5; ++r) {
    CK(hipEventRecord(s));
    for (int i = 0; i < iters; ++i) a.fn(W, xs[i % nx], Y, M, K, N);
    CK(hipEventRecord(e)); CK(hipEventSynchronize(e));
    CK(hipEventElapsedTime(&ms, s, e));
    samp.push_back(ms * 1000.0 / iters);
  }
  std::sort(samp.begin(), samp.end());
  iters_out = iters;
  CK(hipEventDestroy(s)); CK(hipEventDestroy(e));
  return samp[2];  // median us/call
}

// geos, rows; arms[0] is the incumbent-reference for bit checks (full-N).
inline int run_all(const std::vector<Geo>& geos, const std::vector<int>& rowsl, std::vector<Arm>& arms, const char* only_arm, int rounds) {
  hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0));
  printf("# device %s arch %s\n", pr.name, pr.gcnArchName);
  int bad_total = 0;
  for (const Geo& g : geos) {
    const int M = g.M, K = g.K;
    const int NMAX = *std::max_element(rowsl.begin(), rowsl.end());
    Lcg rng{0x5eed0431ULL ^ (uint64_t)M * 7919 + K};
    std::vector<uint16_t> wh((size_t)M * K);
    for (auto& v : wh) v = f2bf(rng.next() * 0.05f);
    uint16_t* dW; CK(hipMalloc(&dW, wh.size() * 2)); CK(hipMemcpy(dW, wh.data(), wh.size() * 2, hipMemcpyHostToDevice));
    size_t xe = g.xbf16 ? 2 : 4;
    size_t xbytes = (size_t)NMAX * K * xe;
    int nx = std::max<int>(2, (int)std::min<size_t>(8, (256u << 20) / xbytes + 1));
    std::vector<void*> xs(nx);
    for (int i = 0; i < nx; ++i) {
      CK(hipMalloc(&xs[i], xbytes));
      if (i == 0 || true) {
        std::vector<uint8_t> hb(xbytes);
        for (size_t j = 0; j < (size_t)NMAX * K; ++j) {
          float v = rng.next() * 2.f; uint16_t b = f2bf(v);
          if (g.xbf16) memcpy(&hb[j * 2], &b, 2); else { float f = bf2f(b); memcpy(&hb[j * 4], &f, 4); }
        }
        CK(hipMemcpy(xs[i], hb.data(), xbytes, hipMemcpyHostToDevice));
      }
    }
    float *dY, *dR; CK(hipMalloc(&dY, (size_t)NMAX * M * 4)); CK(hipMalloc(&dR, (size_t)NMAX * M * 4));
    printf("== geometry %s M=%d K=%d x=%s (X bufs %d x %.1f MB)\n", g.name, M, K, g.xbf16 ? "bf16" : "f32", nx, xbytes / 1e6);
    // bit check vs arms[0] at each rows
    for (int N : rowsl) {
      arms[0].fn(dW, xs[0], dR, M, K, N); CK(hipDeviceSynchronize());
      std::vector<float> ref((size_t)N * M); CK(hipMemcpy(ref.data(), dR, ref.size() * 4, hipMemcpyDeviceToHost));
      for (size_t ai = 1; ai < arms.size(); ++ai) {
        Arm& a = arms[ai];
        if (N < a.minN || N > a.maxN || (a.onlyM && a.onlyM != M)) continue;
        CK(hipMemset(dY, 0xff, (size_t)N * M * 4));
        a.fn(dW, xs[0], dY, M, K, N); CK(hipDeviceSynchronize());
        std::vector<float> got((size_t)N * M); CK(hipMemcpy(got.data(), dY, got.size() * 4, hipMemcpyDeviceToHost));
        size_t bad = 0; for (size_t j = 0; j < got.size(); ++j) if (memcmp(&got[j], &ref[j], 4) != 0) ++bad;
        if (bad) { printf("BITCHECK FAIL geo=%s arm=%s N=%d: %zu/%zu differ\n", g.name, a.name.c_str(), N, bad, got.size()); ++bad_total; }
      }
    }
    // timing
    for (int N : rowsl) {
      double floor_bytes = (double)M * K * 2 + (double)N * K * xe + (double)N * M * 4;
      for (int r = 0; r < rounds; ++r)
        for (auto& a : arms) {
          if (N < a.minN || N > a.maxN || (a.onlyM && a.onlyM != M)) continue;
          if (only_arm && !strstr(a.name.c_str(), only_arm) && &a != &arms[0]) continue;
          int it; double us = time_arm(a, dW, xs.data(), nx, dY, M, K, N, it);
          printf("T geo=%s N=%d arm=%s round=%d us=%.3f GBs=%.1f iters=%d\n", g.name, N, a.name.c_str(), r, us, floor_bytes / us / 1e3, it);
        }
    }
    for (auto p : xs) CK(hipFree(p));
    CK(hipFree(dW)); CK(hipFree(dY)); CK(hipFree(dR));
  }
  printf(bad_total ? "BITCHECK FAILED %d\n" : "BITCHECK OK\n", bad_total);
  return bad_total ? 3 : 0;
}
