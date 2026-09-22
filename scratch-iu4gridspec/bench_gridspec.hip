#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../kernels/src/block_i4_128_quant.hip"
#include "../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip"
#include "../kernels/src/gemm_mq4g256v2_residual_mmq_iu4_gridspec.gfx11.hip"

#define HIP_OK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
  std::exit(1); } } while (0)

constexpr size_t LDS_BYTES = 30720;

__global__ void compare_bits(const float* a, const float* b, size_t n,
                             unsigned long long* mismatches) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && __builtin_bit_cast(unsigned, a[i]) != __builtin_bit_cast(unsigned, b[i]))
    atomicAdd(mismatches, 1ull);
}

__global__ void init_residual(float* y, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = 0.25f;
}

static void configure_kernels() {
  HIP_OK(hipFuncSetAttribute((const void*)gemm_mq4g256v2_residual_mmq_iu4,
      hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES));
  HIP_OK(hipFuncSetAttribute((const void*)gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec,
      hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES));
  HIP_OK(hipFuncSetAttribute((const void*)gemm_mq4g256v2_residual_mmq_iu4_branch_gridspec,
      hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES));
  HIP_OK(hipFuncSetAttribute((const void*)gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151,
      hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES));
  HIP_OK(hipFuncSetAttribute((const void*)gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151,
      hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES));
#if defined(__gfx1151__)
  HIP_OK(hipFuncSetAttribute((const void*)gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151,
      hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES));
#endif
}

static void launch_full(const char* w, const block_i4_128* x, float* y,
                        int M, int K, int N, int add, int col_tiles) {
  const dim3 grid(col_tiles, M / 128);
  if (add) {
    gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151
        <<<grid, dim3(32, 8), LDS_BYTES>>>(w, x, y, M, K, N, add);
  } else {
#if defined(__gfx1151__)
    gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151
        <<<grid, dim3(32, 16), LDS_BYTES>>>(w, x, y, M, K, N, add);
#else
    gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151
        <<<grid, dim3(32, 8), LDS_BYTES>>>(w, x, y, M, K, N, add);
#endif
  }
}

static void launch_full_ordinary(const char* w, const block_i4_128* x, float* y,
                                 int M, int K, int N, int add, int col_tiles) {
  const dim3 grid(col_tiles, M / 128);
  if (add) {
    gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151
        <<<grid, dim3(32, 8), LDS_BYTES>>>(w, x, y, M, K, N, add);
  } else {
    gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3_col_gfx1151
        <<<grid, dim3(32, 8), LDS_BYTES>>>(w, x, y, M, K, N, add);
  }
}

static void launch_shipped(const char* w, const block_i4_128* x, float* y,
                           int M, int K, int N, int add) {
  if (N % 128 == 0) {
    launch_full(w, x, y, M, K, N, add, N / 128);
  } else {
    gemm_mq4g256v2_residual_mmq_iu4
        <<<dim3(M / 128, (N + 127) / 128), dim3(32, 8), LDS_BYTES>>>
        (w, x, y, M, K, N, add);
  }
}

static void launch_split(const char* w, const block_i4_128* x, float* y,
                         int M, int K, int N, int add) {
  if (N % 128 == 0) {
    launch_full(w, x, y, M, K, N, add, N / 128);
    return;
  }
  launch_full(w, x, y, M, K, N, add, N / 128);
  gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec
      <<<dim3(M / 128, 1), dim3(32, 8), LDS_BYTES>>>
      (w, x, y, M, K, N, add);
}

static void launch_split_ordinary(const char* w, const block_i4_128* x, float* y,
                                  int M, int K, int N, int add) {
  if (N % 128 == 0) {
    launch_full(w, x, y, M, K, N, add, N / 128);
    return;
  }
  launch_full_ordinary(w, x, y, M, K, N, add, N / 128);
  gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec
      <<<dim3(M / 128, 1), dim3(32, 8), LDS_BYTES>>>
      (w, x, y, M, K, N, add);
}

static void launch_branch(const char* w, const block_i4_128* x, float* y,
                          int M, int K, int N, int add) {
  if (N % 128 == 0) {
    launch_full(w, x, y, M, K, N, add, N / 128);
    return;
  }
  gemm_mq4g256v2_residual_mmq_iu4_branch_gridspec
      <<<dim3((N + 127) / 128, M / 128), dim3(32, 8), LDS_BYTES>>>
      (w, x, y, M, K, N, add);
}

using Launch = void (*)(const char*, const block_i4_128*, float*, int, int, int, int);

static void run_arm(const char* variant, Launch launch,
                    const char* w, const block_i4_128* x, float* y,
                    int M, int K, int N, int add) {
  for (int i = 0; i < 2; ++i) launch(w, x, y, M, K, N, add);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> us;
  for (int i = 0; i < 9; ++i) {
    hipEvent_t a, b;
    HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
    HIP_OK(hipEventRecord(a));
    launch(w, x, y, M, K, N, add);
    HIP_OK(hipEventRecord(b)); HIP_OK(hipEventSynchronize(b));
    float ms = 0.0f; HIP_OK(hipEventElapsedTime(&ms, a, b));
    us.push_back(ms * 1000.0f);
    HIP_OK(hipEventDestroy(a)); HIP_OK(hipEventDestroy(b));
  }
  std::sort(us.begin(), us.end());
  const double median = us[us.size() / 2];
  const double tops = 2.0 * (double)M * K * N / (median * 1.0e6);
  std::printf("RESULT,variant=%s,shape=%s,N=%d,M=%d,K=%d,add=%d,median_us=%.3f,tops=%.3f,spread_pct=%.3f\n",
      variant, add ? "down" : "gate", N, M, K, add, median, tops,
      100.0 * (us.back() - us.front()) / median);
}

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::fprintf(stderr,
        "usage: %s gate|down N [--reverse] [--branch|--ordinary-split]\n", argv[0]);
    return 2;
  }
  const bool down = std::strcmp(argv[1], "down") == 0;
  const int N = std::atoi(argv[2]);
  bool reverse = false;
  bool branch = false;
  bool ordinary_split = false;
  for (int i = 3; i < argc; ++i) {
    reverse |= std::strcmp(argv[i], "--reverse") == 0;
    branch |= std::strcmp(argv[i], "--branch") == 0;
    ordinary_split |= std::strcmp(argv[i], "--ordinary-split") == 0;
  }
  const Launch candidate =
      ordinary_split ? launch_split_ordinary : (branch ? launch_branch : launch_split);
  const char* candidate_name = ordinary_split
      ? "gridspec_split_ordinary"
      : (branch ? "gridspec_branch" : "gridspec_split");
  const int M = down ? 5120 : 17408;
  const int K = down ? 17408 : 5120;
  const int add = down ? 1 : 0;

  hipDeviceProp_t prop{}; HIP_OK(hipGetDeviceProperties(&prop, 0));
  const char* expected = std::getenv("EXPECT_ARCH");
  if (!expected || std::strstr(prop.gcnArchName, expected) == nullptr) {
    std::fprintf(stderr, "ARCH_ASSERT expected=%s actual=%s\n",
        expected ? expected : "<unset>", prop.gcnArchName);
    return 3;
  }
  const char* mode = ordinary_split ? "ordinary-split" : (branch ? "branch" : "split");
  std::printf("DEVICE,arch=%s,name=%s,order=%s,mode=%s\n", prop.gcnArchName, prop.name,
      reverse ? "reverse" : "forward", mode);

  const size_t groups = (size_t)M * (K / 256);
  const size_t blocks = (size_t)(K / 128) * N;
  const size_t elems = (size_t)N * M;
  std::vector<unsigned char> hw(groups * 136);
  for (size_t g = 0; g < groups; ++g) {
    const __half sc = (__half)(0.005f + 0.0001f * (float)(g % 13));
    const __half zp = (__half)(-0.04f + 0.0001f * (float)(g % 7));
    std::memcpy(hw.data() + g * 136 + 0, &sc, sizeof(sc));
    std::memcpy(hw.data() + g * 136 + 2, &zp, sizeof(zp));
    std::memcpy(hw.data() + g * 136 + 4, &sc, sizeof(sc));
    std::memcpy(hw.data() + g * 136 + 6, &zp, sizeof(zp));
    for (int q = 8; q < 136; ++q)
      hw[g * 136 + q] = (unsigned char)((g * 17 + q * 29) & 255);
  }
  std::vector<block_i4_128> hx(blocks);
  for (size_t b = 0; b < blocks; ++b) {
    hx[b].d = 0.01f + 0.0001f * (float)(b % 11);
    hx[b].s = (int)(b % 31) - 15;
    for (int q = 0; q < 64; ++q)
      hx[b].qs[q] = (unsigned char)((b * 23 + q * 31) & 255);
  }

  char* w = nullptr; block_i4_128* x = nullptr;
  float *ya = nullptr, *yb = nullptr;
  unsigned long long* mismatches = nullptr;
  HIP_OK(hipMalloc(&w, hw.size())); HIP_OK(hipMalloc(&x, hx.size() * sizeof(block_i4_128)));
  HIP_OK(hipMalloc(&ya, elems * sizeof(float))); HIP_OK(hipMalloc(&yb, elems * sizeof(float)));
  HIP_OK(hipMalloc(&mismatches, sizeof(*mismatches)));
  HIP_OK(hipMemcpy(w, hw.data(), hw.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(x, hx.data(), hx.size() * sizeof(block_i4_128), hipMemcpyHostToDevice));
  configure_kernels();

  if (add) {
    init_residual<<<(elems + 255) / 256, 256>>>(ya, elems);
    init_residual<<<(elems + 255) / 256, 256>>>(yb, elems);
  } else {
    HIP_OK(hipMemset(ya, 0, elems * sizeof(float)));
    HIP_OK(hipMemset(yb, 0, elems * sizeof(float)));
  }
  launch_shipped(w, x, ya, M, K, N, add);
  candidate(w, x, yb, M, K, N, add);
  HIP_OK(hipMemset(mismatches, 0, sizeof(*mismatches)));
  compare_bits<<<(elems + 255) / 256, 256>>>(ya, yb, elems, mismatches);
  unsigned long long host_mismatches = 0;
  HIP_OK(hipMemcpy(&host_mismatches, mismatches, sizeof(host_mismatches), hipMemcpyDeviceToHost));
  std::printf("PARITY,shape=%s,N=%d,mismatches=%llu,total=%zu\n",
      down ? "down" : "gate", N, host_mismatches, elems);
  if (host_mismatches != 0) return 4;

  if (!reverse) {
    run_arm("shipped", launch_shipped, w, x, ya, M, K, N, add);
    run_arm(candidate_name, candidate, w, x, yb, M, K, N, add);
  } else {
    run_arm(candidate_name, candidate, w, x, yb, M, K, N, add);
    run_arm("shipped", launch_shipped, w, x, ya, M, K, N, add);
  }

  HIP_OK(hipFree(mismatches)); HIP_OK(hipFree(yb)); HIP_OK(hipFree(ya));
  HIP_OK(hipFree(x)); HIP_OK(hipFree(w));
  return 0;
}
