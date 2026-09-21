// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef BENCH_FAMILY
#define BENCH_FAMILY 0
#endif

#if BENCH_FAMILY == 0
#define HIPFIRE_FP8_QKVZA 0
#define HIPFIRE_FP8_RESIDUAL 0
#define HIPFIRE_FP8_QKV 0
#define HIPFIRE_FP8_FOLDFREE_KERNEL gemm_gate_up_mq4g256v2_wmma_fp8_v2_foldfree_gfx1201
#elif BENCH_FAMILY == 1
#define HIPFIRE_FP8_QKVZA 0
#define HIPFIRE_FP8_RESIDUAL 1
#define HIPFIRE_FP8_QKV 0
#define HIPFIRE_FP8_FOLDFREE_KERNEL gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_gfx1201
#elif BENCH_FAMILY == 2
#define HIPFIRE_FP8_QKVZA 1
#define HIPFIRE_FP8_RESIDUAL 0
#define HIPFIRE_FP8_QKV 0
#define HIPFIRE_FP8_FOLDFREE_KERNEL gemm_qkvza_mq4g256v2_wmma_fp8_v2_foldfree_gfx1201
#elif BENCH_FAMILY == 3
#define HIPFIRE_FP8_QKVZA 0
#define HIPFIRE_FP8_RESIDUAL 0
#define HIPFIRE_FP8_QKV 1
#define HIPFIRE_FP8_FOLDFREE_KERNEL gemm_qkv_mq4g256v2_wmma_fp8_v2_foldfree_gfx1201
#else
#error invalid BENCH_FAMILY
#endif

#include "../kernels/src/gemm_mq4g256v2_wmma_fp8_v2_foldfree_common.gfx12.hip"

#define HIP_OK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); \
} } while (0)

static constexpr const char* kArtifact =
    "/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-pow2h-a035.hfq";
static constexpr size_t kLdsBytes = 18944;

struct WeightSpec { const char* name; uint64_t offset; int m; int k; };
#if BENCH_FAMILY == 0
static constexpr WeightSpec kSpecs[] = {
  {"gate", 2149750720ull, 17408, 5120}, {"up", 2197110720ull, 17408, 5120},
};
#elif BENCH_FAMILY == 1
static constexpr WeightSpec kSpecs[] = {{"residual", 2102366144ull, 5120, 17408}};
#elif BENCH_FAMILY == 2
static constexpr WeightSpec kSpecs[] = {
  {"qkv", 2041056960ull, 10240, 5120}, {"z", 2068920000ull, 6144, 5120},
  {"beta", 2040775360ull, 48, 5120}, {"alpha", 2040916160ull, 48, 5120},
};
#else
static constexpr WeightSpec kSpecs[] = {
  {"q", 2813645632ull, 12288, 5120}, {"k", 2794125632ull, 1024, 5120},
  {"v", 2847079232ull, 1024, 5120},
};
#endif

struct Weight {
  int m = 0, k = 0;
  std::vector<uint8_t> host;
  std::vector<float> erow;
  char* dev = nullptr;
  float* dev_erow = nullptr;
};

static void pread_all(int fd, void* dst, size_t bytes, uint64_t offset) {
  auto* p = static_cast<uint8_t*>(dst);
  while (bytes) {
    const ssize_t got = pread(fd, p, bytes, static_cast<off_t>(offset));
    if (got <= 0) { std::perror("pread"); std::exit(2); }
    p += got; bytes -= static_cast<size_t>(got); offset += static_cast<size_t>(got);
  }
}

static Weight load_weight(int fd, const WeightSpec& spec, int rows) {
  Weight w; w.m = rows; w.k = spec.k;
  const size_t gpr = static_cast<size_t>(spec.k) / 256;
  const size_t row_bytes = gpr * 136;
  w.host.resize(static_cast<size_t>(rows) * row_bytes);
  pread_all(fd, w.host.data(), w.host.size(), spec.offset);
  w.erow.resize(rows);
  for (int r = 0; r < rows; ++r) {
    unsigned max_exp = 0;
    for (size_t g = 0; g < gpr; ++g) {
      const uint8_t* h = w.host.data() + (static_cast<size_t>(r) * gpr + g) * 136;
      for (int off : {0, 4}) {
        const uint16_t bits = static_cast<uint16_t>(h[off]) |
                              (static_cast<uint16_t>(h[off + 1]) << 8);
        const unsigned exp = (bits >> 10) & 31;
        const unsigned mant = bits & 0x3ff;
        if ((bits & 0x8000) || exp == 0 || exp == 31 || (mant != 0 && mant != 0x200)) {
          std::fprintf(stderr, "%s row %d invalid pow2h d=0x%04x\n", spec.name, r, bits);
          std::exit(2);
        }
        max_exp = std::max(max_exp, exp);
      }
    }
    w.erow[r] = std::ldexp(1.0f, static_cast<int>(max_exp) - 15);
  }
  HIP_OK(hipMalloc(&w.dev, w.host.size()));
  HIP_OK(hipMalloc(&w.dev_erow, w.erow.size() * sizeof(float)));
  HIP_OK(hipMemcpy(w.dev, w.host.data(), w.host.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(w.dev_erow, w.erow.data(), w.erow.size() * sizeof(float), hipMemcpyHostToDevice));
  return w;
}

static float fp8_value(uint8_t b) {
  const bool neg = b & 0x80;
  const unsigned exp = (b >> 3) & 15;
  const unsigned mant = b & 7;
  float v;
  if (exp == 0) v = std::ldexp(static_cast<float>(mant), -9);
  else if (exp == 15 && mant == 7) return NAN;
  else v = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f,
                      exp == 15 ? 8 : static_cast<int>(exp) - 7);
  return neg ? -v : v;
}

static uint8_t fp8_rne(float x) {
  if (x == 0.0f) return 0;
  const bool neg = std::signbit(x);
  const float ax = std::fabs(x);
  float best = INFINITY;
  uint8_t best_b = 0;
  for (unsigned b = 0; b <= 0x7e; ++b) {
    const float v = fp8_value(static_cast<uint8_t>(b));
    const float d = std::fabs(v - ax);
    if (d < best || (d == best && (b & 1) == 0 && (best_b & 1) != 0)) {
      best = d; best_b = static_cast<uint8_t>(b);
    }
  }
  return neg ? static_cast<uint8_t>(best_b | 0x80) : best_b;
}

static float folded_weight(const Weight& w, int row, int col) {
  const int gpr = w.k / 256;
  const int group = col / 256;
  const int in_group = col & 255;
  const uint8_t* h = w.host.data() + (static_cast<size_t>(row) * gpr + group) * 136;
  const int half = in_group >> 7;
  const uint16_t bits = static_cast<uint16_t>(h[half * 4]) |
                        (static_cast<uint16_t>(h[half * 4 + 1]) << 8);
  const int e = static_cast<int>((bits >> 10) & 31) - 15;
  const float mantissa = (bits & 0x3ff) == 0x200 ? 1.5f : 1.0f;
  const uint8_t packed = h[8 + in_group / 2];
  const int code = (in_group & 1) ? packed >> 4 : packed & 15;
  const float first = fp8_value(fp8_rne((static_cast<float>(code) - 8.0f) * mantissa));
  int erow_exp = 0; std::frexp(w.erow[row], &erow_exp); --erow_exp;
  return fp8_value(fp8_rne(std::ldexp(first, e - erow_exp)));
}

static std::vector<uint8_t> make_x(int n, int k) {
  static constexpr uint8_t values[] = {
    0x00,0x30,0x38,0x3c,0x40,0x44,0xb0,0xb8,0xbc,0xc0,0x28,0xa8,0x42,0xc2,0x34,0xb4
  };
  std::vector<uint8_t> x(static_cast<size_t>(n) * k);
  for (int t = 0; t < n; ++t)
    for (int c = 0; c < k; ++c)
      x[static_cast<size_t>(t) * k + c] = values[(t * 13 + c * 7 + (c >> 5)) & 15];
  return x;
}

static void launch(dim3 grid, dim3 block, size_t lds, hipStream_t stream,
                   const std::vector<Weight>& w, uint8_t* x, float* hs, float* xs,
                   const std::vector<float*>& y, const std::vector<int>& m, int k, int n) {
#if BENCH_FAMILY == 0
  hipLaunchKernelGGL(HIPFIRE_FP8_FOLDFREE_KERNEL, grid, block, lds, stream,
    w[0].dev, w[1].dev, w[0].dev_erow, w[1].dev_erow, x, hs, xs,
    y[0], y[1], m[0], m[1], k, n);
#elif BENCH_FAMILY == 1
  hipLaunchKernelGGL(HIPFIRE_FP8_FOLDFREE_KERNEL, grid, block, lds, stream,
    w[0].dev, w[0].dev_erow, x, hs, xs, y[0], m[0], k, n);
#elif BENCH_FAMILY == 2
  hipLaunchKernelGGL(HIPFIRE_FP8_FOLDFREE_KERNEL, grid, block, lds, stream,
    w[0].dev, w[1].dev, w[2].dev, w[3].dev,
    w[0].dev_erow, w[1].dev_erow, w[2].dev_erow, w[3].dev_erow,
    x, hs, xs, y[0], y[1], y[2], y[3], m[0], m[1], m[2], m[3], k, n);
#else
  hipLaunchKernelGGL(HIPFIRE_FP8_FOLDFREE_KERNEL, grid, block, lds, stream,
    w[0].dev, w[1].dev, w[2].dev, w[0].dev_erow, w[1].dev_erow, w[2].dev_erow,
    x, hs, xs, y[0], y[1], y[2], m[0], m[1], m[2], k, n);
#endif
}

int main(int argc, char** argv) {
  const bool exact = argc > 1 && std::string(argv[1]) == "exact";
  const int fd = open(kArtifact, O_RDONLY);
  if (fd < 0) { std::perror("open artifact"); return 2; }
#if BENCH_FAMILY == 0
  std::vector<int> m = exact ? std::vector<int>{64,64} : std::vector<int>{17408,17408};
#elif BENCH_FAMILY == 1
  std::vector<int> m = exact ? std::vector<int>{128} : std::vector<int>{5120};
#elif BENCH_FAMILY == 2
  std::vector<int> m = exact ? std::vector<int>{32,32,32,32} : std::vector<int>{10240,6144,48,48};
#else
  std::vector<int> m = exact ? std::vector<int>{64,32,32} : std::vector<int>{12288,1024,1024};
#endif
  const int k = kSpecs[0].k;
  const int n = exact ? 256 : 8192;
  std::vector<Weight> weights;
  for (size_t i = 0; i < sizeof(kSpecs)/sizeof(kSpecs[0]); ++i)
    weights.push_back(load_weight(fd, kSpecs[i], m[i]));
  close(fd);

  std::vector<uint8_t> hx = make_x(n, k);
  std::vector<float> hxs(n);
  for (int t = 0; t < n; ++t) hxs[t] = std::ldexp(1.0f, (t % 5) - 2);
  uint8_t* dx = nullptr; float* dhs = nullptr; float* dxs = nullptr;
  HIP_OK(hipMalloc(&dx, hx.size()));
  HIP_OK(hipMalloc(&dhs, static_cast<size_t>(n) * (k / 256) * 2 * sizeof(float)));
  HIP_OK(hipMalloc(&dxs, hxs.size() * sizeof(float)));
  HIP_OK(hipMemcpy(dx, hx.data(), hx.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemset(dhs, 0, static_cast<size_t>(n) * (k / 256) * 2 * sizeof(float)));
  HIP_OK(hipMemcpy(dxs, hxs.data(), hxs.size() * sizeof(float), hipMemcpyHostToDevice));

  std::vector<float*> dy(m.size(), nullptr);
  std::vector<std::vector<float>> initial(m.size());
  for (size_t i = 0; i < m.size(); ++i) {
    const size_t elems = static_cast<size_t>(n) * m[i];
    HIP_OK(hipMalloc(&dy[i], elems * sizeof(float)));
    initial[i].assign(elems, BENCH_FAMILY == 1 ? 0.125f : 0.0f);
    HIP_OK(hipMemcpy(dy[i], initial[i].data(), elems * sizeof(float), hipMemcpyHostToDevice));
  }
  int total_m = 0; for (int v : m) total_m += v;
  const dim3 grid((total_m + 127) / 128, (n + 255) / 256, 1), block(256,1,1);
  int blocks = 0;
  HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, HIPFIRE_FP8_FOLDFREE_KERNEL, 256, kLdsBytes));

  if (exact) {
    std::vector<std::vector<float>> folded(m.size());
    for (size_t part = 0; part < m.size(); ++part) {
      folded[part].resize(static_cast<size_t>(m[part]) * k);
      for (int r = 0; r < m[part]; ++r)
        for (int c = 0; c < k; ++c)
          folded[part][static_cast<size_t>(r) * k + c] =
              folded_weight(weights[part], r, c);
    }
    launch(grid, block, kLdsBytes, nullptr, weights, dx, dhs, dxs, dy, m, k, n);
    HIP_OK(hipDeviceSynchronize());
    double err2 = 0.0, ref2 = 0.0, max_abs = 0.0; size_t bad = 0, count = 0;
    for (size_t part = 0; part < m.size(); ++part) {
      std::vector<float> got(static_cast<size_t>(n) * m[part]);
      HIP_OK(hipMemcpy(got.data(), dy[part], got.size() * sizeof(float), hipMemcpyDeviceToHost));
      for (int t = 0; t < n; ++t) {
        for (int r = 0; r < m[part]; ++r) {
          float sum = 0.0f;
          for (int c = 0; c < k; ++c) {
            const float av = fp8_value(hx[static_cast<size_t>(t) * k + c]);
            sum = std::fma(av, folded[part][static_cast<size_t>(r) * k + c], sum);
          }
          const float ref = initial[part][static_cast<size_t>(t) * m[part] + r]
                          + hxs[t] * weights[part].erow[r] * sum;
          const float gv = got[static_cast<size_t>(t) * m[part] + r];
          if (!std::isfinite(gv) || !std::isfinite(ref)) ++bad;
          const double d = static_cast<double>(gv) - ref;
          err2 += d*d; ref2 += static_cast<double>(ref)*ref;
          max_abs = std::max(max_abs, std::fabs(d)); ++count;
        }
      }
    }
    const double rel_rms = std::sqrt(err2 / std::max(ref2, 1e-300));
    std::printf("family=%d exact rows=%d n=%d k=%d rel_rms=%.9g max_abs=%.9g nonfinite=%zu blocks_per_cu=%d\n",
                BENCH_FAMILY, total_m, n, k, rel_rms, max_abs, bad, blocks);
    return (bad == 0 && rel_rms <= 1e-6) ? 0 : 1;
  }

  for (int i = 0; i < 5; ++i) launch(grid, block, kLdsBytes, nullptr, weights, dx, dhs, dxs, dy, m, k, n);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> us;
  for (int i = 0; i < 20; ++i) {
    hipEvent_t a, b; HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
    HIP_OK(hipEventRecord(a));
    launch(grid, block, kLdsBytes, nullptr, weights, dx, dhs, dxs, dy, m, k, n);
    HIP_OK(hipEventRecord(b)); HIP_OK(hipEventSynchronize(b));
    float ms = 0; HIP_OK(hipEventElapsedTime(&ms, a, b)); us.push_back(ms * 1000.0f);
    HIP_OK(hipEventDestroy(a)); HIP_OK(hipEventDestroy(b));
  }
  std::sort(us.begin(), us.end());
  const double median_us = (us[9] + us[10]) * 0.5;
  const double flop = 2.0 * n * static_cast<double>(k) * total_m;
  std::printf("family=%d m=%d n=%d k=%d median_us=%.3f tflops=%.3f blocks_per_cu=%d lds=%zu\n",
              BENCH_FAMILY, total_m, n, k, median_us, flop / median_us / 1e6, blocks, kLdsBytes);
  return 0;
}
