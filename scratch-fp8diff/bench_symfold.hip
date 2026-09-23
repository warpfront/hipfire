#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef BENCH_BM
#define BENCH_BM 128
#endif
#ifndef BENCH_BN
#define BENCH_BN 128
#endif
#ifndef BENCH_RESIDUAL
#define BENCH_RESIDUAL 0
#endif

#define HIPFIRE_FP8_SYMFOLD 1
#define HIPFIRE_FP8_V2_TILE 1
#define HIPFIRE_FP8_V2_BM BENCH_BM
#define HIPFIRE_FP8_V2_BN BENCH_BN
#define HIPFIRE_FP8_V2_BK 64
#define HIPFIRE_FP8_V2_WAVES 8
#define HIPFIRE_FP8_RESIDUAL BENCH_RESIDUAL
#define HIPFIRE_FP8_GATEUP_KERNEL bench_symfold
#include "/home/kaden/ClaudeCode/warpfront/wt-fp8sym/kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip"

#define HIP_OK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); \
} } while (0)

static void* zalloc(size_t bytes) {
  void* p = nullptr;
  HIP_OK(hipMalloc(&p, bytes));
  HIP_OK(hipMemset(p, 0, bytes));
  return p;
}

static double timed_20(void (*launch)(void*), void* ctx, double* mean_us) {
  for (int i = 0; i < 5; ++i) launch(ctx);
  HIP_OK(hipDeviceSynchronize());
  std::vector<hipEvent_t> starts(20), stops(20);
  for (int i = 0; i < 20; ++i) {
    HIP_OK(hipEventCreate(&starts[i]));
    HIP_OK(hipEventCreate(&stops[i]));
    HIP_OK(hipEventRecord(starts[i], 0));
    launch(ctx);
    HIP_OK(hipGetLastError());
    HIP_OK(hipEventRecord(stops[i], 0));
  }
  HIP_OK(hipEventSynchronize(stops.back()));
  std::vector<double> us;
  double sum = 0.0;
  for (int i = 0; i < 20; ++i) {
    float ms = 0.0f;
    HIP_OK(hipEventElapsedTime(&ms, starts[i], stops[i]));
    us.push_back((double)ms * 1000.0);
    sum += us.back();
    HIP_OK(hipEventDestroy(starts[i]));
    HIP_OK(hipEventDestroy(stops[i]));
  }
  std::sort(us.begin(), us.end());
  *mean_us = sum / 20.0;
  return 0.5 * (us[9] + us[10]);
}

constexpr int BATCH = 8192;
constexpr int LDS_BYTES = (BENCH_BM + BENCH_BN) * 72 + BENCH_BN * 4;

#if BENCH_RESIDUAL
struct Args {
  const char* w; const unsigned char* x; const float* sums; const float* scales;
  float* y; int rows; int k; dim3 grid;
};
static void launch_kernel(void* p) {
  Args& a = *static_cast<Args*>(p);
  hipLaunchKernelGGL(bench_symfold, a.grid, dim3(256), LDS_BYTES, 0,
                     a.w, a.x, a.sums, a.scales, a.y, a.rows, a.k, BATCH);
}
static void run() {
  constexpr int ROWS = 5120, K = 17408;
  size_t gpr = K / 256;
  auto* w = static_cast<char*>(zalloc((size_t)ROWS * gpr * 136));
  auto* x = static_cast<unsigned char*>(zalloc((size_t)BATCH * K));
  auto* sums = static_cast<float*>(zalloc((size_t)BATCH * gpr * 2 * sizeof(float)));
  auto* scales = static_cast<float*>(zalloc((size_t)BATCH * sizeof(float)));
  auto* y = static_cast<float*>(zalloc((size_t)BATCH * ROWS * sizeof(float)));
  Args a{w, x, sums, scales, y, ROWS, K,
         dim3((ROWS + BENCH_BN - 1) / BENCH_BN, (BATCH + BENCH_BM - 1) / BENCH_BM, 1)};
  double mean_us = 0.0;
  double median_us = timed_20(launch_kernel, &a, &mean_us);
  double tf = 2.0 * BATCH * (double)ROWS * K / median_us / 1.0e6;
  int occ = 0;
  HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, bench_symfold, 256, LDS_BYTES));
  std::printf("RESULT,hipfire_symfold,down,BM=%d,BN=%d,M=%d,N=%d,K=%d,grid=%u:%u,LDS=%d,median_us=%.3f,mean_us=%.3f,tflops=%.3f,occ_blocks_cu=%d\n",
              BENCH_BM, BENCH_BN, BATCH, ROWS, K, a.grid.x, a.grid.y, LDS_BYTES, median_us, mean_us, tf, occ);
  HIP_OK(hipFree(w)); HIP_OK(hipFree(x)); HIP_OK(hipFree(sums)); HIP_OK(hipFree(scales)); HIP_OK(hipFree(y));
}
#else
struct Args {
  const char* wg; const char* wu; const unsigned char* x; const float* sums; const float* scales;
  float* yg; float* yu; int rows; int k; dim3 grid;
};
static void launch_kernel(void* p) {
  Args& a = *static_cast<Args*>(p);
  hipLaunchKernelGGL(bench_symfold, a.grid, dim3(256), LDS_BYTES, 0,
                     a.wg, a.wu, a.x, a.sums, a.scales, a.yg, a.yu,
                     a.rows, a.rows, a.k, BATCH);
}
static void run() {
  constexpr int ROWS = 17408, K = 5120;
  size_t gpr = K / 256;
  auto* wg = static_cast<char*>(zalloc((size_t)ROWS * gpr * 136));
  auto* wu = static_cast<char*>(zalloc((size_t)ROWS * gpr * 136));
  auto* x = static_cast<unsigned char*>(zalloc((size_t)BATCH * K));
  auto* sums = static_cast<float*>(zalloc((size_t)BATCH * gpr * 2 * sizeof(float)));
  auto* scales = static_cast<float*>(zalloc((size_t)BATCH * sizeof(float)));
  auto* yg = static_cast<float*>(zalloc((size_t)BATCH * ROWS * sizeof(float)));
  auto* yu = static_cast<float*>(zalloc((size_t)BATCH * ROWS * sizeof(float)));
  Args a{wg, wu, x, sums, scales, yg, yu, ROWS, K,
         dim3((2 * ROWS + BENCH_BN - 1) / BENCH_BN, (BATCH + BENCH_BM - 1) / BENCH_BM, 1)};
  double mean_us = 0.0;
  double median_us = timed_20(launch_kernel, &a, &mean_us);
  double tf = 2.0 * BATCH * (double)(2 * ROWS) * K / median_us / 1.0e6;
  int occ = 0;
  HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, bench_symfold, 256, LDS_BYTES));
  std::printf("RESULT,hipfire_symfold,gate_up,BM=%d,BN=%d,M=%d,N=%d,K=%d,grid=%u:%u,LDS=%d,median_us=%.3f,mean_us=%.3f,tflops=%.3f,occ_blocks_cu=%d\n",
              BENCH_BM, BENCH_BN, BATCH, 2 * ROWS, K, a.grid.x, a.grid.y, LDS_BYTES, median_us, mean_us, tf, occ);
  HIP_OK(hipFree(wg)); HIP_OK(hipFree(wu)); HIP_OK(hipFree(x)); HIP_OK(hipFree(sums)); HIP_OK(hipFree(scales)); HIP_OK(hipFree(yg)); HIP_OK(hipFree(yu));
}
#endif

int main() {
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  std::printf("DEVICE,%s,%s\n", prop.name, prop.gcnArchName);
  run();
  return 0;
}
