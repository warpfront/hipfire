#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "radiance_mxfp4_fp8.hip"

#define HIP_OK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); \
} } while (0)

static void* zalloc(size_t bytes) {
  void* p = nullptr;
  HIP_OK(hipMalloc(&p, bytes));
  HIP_OK(hipMemset(p, 0, bytes));
  return p;
}

static void bench(const char* label, int M, int N, int K) {
  auto* AT = static_cast<unsigned char*>(zalloc((size_t)M * K));
  auto* W = static_cast<unsigned char*>(zalloc((size_t)N * K / 2));
  auto* Ws = static_cast<unsigned char*>(zalloc((size_t)(K / 32) * N));
  auto* Wref = static_cast<unsigned char*>(zalloc((size_t)N));
  auto* As = static_cast<float*>(zalloc((size_t)M * sizeof(float)));
  auto* C = static_cast<__bf16*>(zalloc((size_t)M * N * sizeof(__bf16)));

  dim3 block(256, 1, 1);
  dim3 grid((N + 127) / 128, (M + 255) / 256, 1);
  auto launch = [&]() {
    hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm_atiled<4, true, 64>), grid, block, 0, 0,
                       AT, W, Ws, Wref, As, C, M, N, K, 1 << 30, 1 << 30, 0L);
  };

  for (int i = 0; i < 5; ++i) launch();
  HIP_OK(hipDeviceSynchronize());
  std::vector<hipEvent_t> starts(20), stops(20);
  for (int i = 0; i < 20; ++i) {
    HIP_OK(hipEventCreate(&starts[i]));
    HIP_OK(hipEventCreate(&stops[i]));
    HIP_OK(hipEventRecord(starts[i], 0));
    launch();
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
  double median_us = 0.5 * (us[9] + us[10]);
  double mean_us = sum / 20.0;
  double tflops = 2.0 * (double)M * (double)N * (double)K / median_us / 1.0e6;
  int occ = 0;
  HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, radiance_mxfp4_fp8_gemm_atiled<4, true, 64>, 256, 0));
  std::printf("RESULT,radiance,%s,M=%d,N=%d,K=%d,grid=%u:%u,median_us=%.3f,mean_us=%.3f,tflops=%.3f,occ_blocks_cu=%d\n",
              label, M, N, K, grid.x, grid.y, median_us, mean_us, tflops, occ);

  HIP_OK(hipFree(AT)); HIP_OK(hipFree(W)); HIP_OK(hipFree(Ws));
  HIP_OK(hipFree(Wref)); HIP_OK(hipFree(As)); HIP_OK(hipFree(C));
}

int main() {
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  std::printf("DEVICE,%s,%s\n", prop.name, prop.gcnArchName);
  bench("gate_up", 8192, 34816, 5120);
  bench("down", 8192, 5120, 17408);
  return 0;
}
