#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using i32x2 = int __attribute__((ext_vector_type(2)));
using i32x8 = int __attribute__((ext_vector_type(8)));
using f16x8 = _Float16 __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));

constexpr int kThreads = 256;
constexpr int kWavesPerBlock = kThreads / 32;
constexpr int kChains = 8;
constexpr int kIters = 10000;
constexpr int kTimedLaunches = 7;

#define HIP_OK(call)                                                           \
  do {                                                                         \
    const hipError_t error = (call);                                            \
    if (error != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,       \
                   #call, hipGetErrorString(error));                            \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

extern "C" __global__ __launch_bounds__(kThreads) void
peak_f16(unsigned long long* out, int iters) {
  const f16x8 a = {(_Float16)1.0f,  (_Float16)1.125f, (_Float16)1.25f,
                   (_Float16)1.375f, (_Float16)1.5f,   (_Float16)1.625f,
                   (_Float16)1.75f,  (_Float16)1.875f};
  const f16x8 b = {(_Float16)2.0f,  (_Float16)2.25f, (_Float16)2.5f,
                   (_Float16)2.75f, (_Float16)3.0f,  (_Float16)3.25f,
                   (_Float16)3.5f,  (_Float16)3.75f};
  const f32x8 zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  f32x8 acc[kChains];
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) acc[chain] = zero;

  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
      acc[chain] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
          a, b, acc[chain]);
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) {
    sum += acc[chain][threadIdx.x & 7];
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(out, (unsigned long long)sum);
}

extern "C" __global__ __launch_bounds__(kThreads) void
peak_iu8(unsigned long long* out, int iters) {
  const unsigned lane_bits = threadIdx.x & 7u;
  const i32x2 a = {(int)(0x38383838u | lane_bits),
                   (int)(0x40404040u | lane_bits)};
  const i32x2 b = {(int)(0x42424242u | lane_bits),
                   (int)(0x44444444u | lane_bits)};
  const i32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
  i32x8 acc[kChains];
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) acc[chain] = zero;

  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
      acc[chain] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
          true, a, true, b, acc[chain], false);
    }
  }

  unsigned long long sum = 0;
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) {
    sum += (unsigned)acc[chain][threadIdx.x & 7];
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(out, sum);
}

extern "C" __global__ __launch_bounds__(kThreads) void
peak_fp8(unsigned long long* out, int iters) {
  // Use exactly the same packed operand bits as iu8.
  const unsigned lane_bits = threadIdx.x & 7u;
  const i32x2 a = {(int)(0x38383838u | lane_bits),
                   (int)(0x40404040u | lane_bits)};
  const i32x2 b = {(int)(0x42424242u | lane_bits),
                   (int)(0x44444444u | lane_bits)};
  const f32x8 zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  f32x8 acc[kChains];
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) acc[chain] = zero;

  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
      acc[chain] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
          a, b, acc[chain]);
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) {
    sum += acc[chain][threadIdx.x & 7];
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(out, (unsigned long long)sum);
}

extern "C" __global__ __launch_bounds__(kThreads) void
peak_iu4(unsigned long long* out, int iters) {
  // Match the operand construction of the prior 695.085-TOPS anchor.
  const i32x2 a = {(int)(threadIdx.x | 0x11111111u),
                   (int)(threadIdx.x | 0x22222222u)};
  const i32x2 b = {(int)(threadIdx.x | 0x33333333u),
                   (int)(threadIdx.x | 0x44444444u)};
  const i32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
  i32x8 acc[kChains];
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) acc[chain] = zero;

  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
      acc[chain] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
          true, a, true, b, acc[chain], false);
    }
  }

  unsigned long long sum = 0;
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) {
    sum += (unsigned)acc[chain][threadIdx.x & 7];
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(out, sum);
}

using Kernel = void (*)(unsigned long long*, int);

struct Variant {
  const char* name;
  Kernel kernel;
  int ops_per_instruction;
};

struct Occupancy {
  int blocks_per_cu;
  size_t dynamic_lds;
};

static void launch(Kernel kernel, int blocks, size_t dynamic_lds,
                   unsigned long long* out, int iters = kIters) {
  void* args[] = {&out, &iters};
  HIP_OK(hipLaunchKernel((const void*)kernel, dim3(blocks), dim3(kThreads),
                         args, dynamic_lds, nullptr));
}

static void run_cell(const Variant& variant, const Occupancy& requested, int cu,
                     unsigned long long* out) {
  HIP_OK(hipFuncSetAttribute((const void*)variant.kernel,
                             hipFuncAttributeMaxDynamicSharedMemorySize,
                             (int)requested.dynamic_lds));
  int available = 0;
  HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &available, (const void*)variant.kernel, kThreads, requested.dynamic_lds));
  if (available != requested.blocks_per_cu) {
    std::fprintf(stderr,
                 "occupancy mismatch for %s: requested=%d available=%d lds=%zu\n",
                 variant.name, requested.blocks_per_cu, available,
                 requested.dynamic_lds);
    std::exit(2);
  }

  const int blocks = cu * requested.blocks_per_cu;
  launch(variant.kernel, blocks, requested.dynamic_lds, out);
  HIP_OK(hipDeviceSynchronize());

  std::vector<float> milliseconds;
  milliseconds.reserve(kTimedLaunches);
  for (int sample = 0; sample < kTimedLaunches; ++sample) {
    hipEvent_t begin;
    hipEvent_t end;
    HIP_OK(hipEventCreate(&begin));
    HIP_OK(hipEventCreate(&end));
    HIP_OK(hipEventRecord(begin));
    launch(variant.kernel, blocks, requested.dynamic_lds, out);
    HIP_OK(hipEventRecord(end));
    HIP_OK(hipEventSynchronize(end));
    float elapsed = 0.0f;
    HIP_OK(hipEventElapsedTime(&elapsed, begin, end));
    milliseconds.push_back(elapsed);
    HIP_OK(hipEventDestroy(begin));
    HIP_OK(hipEventDestroy(end));
  }
  std::sort(milliseconds.begin(), milliseconds.end());
  const double median_ms = milliseconds[kTimedLaunches / 2];
  const double operations =
      (double)blocks * kWavesPerBlock * kIters * kChains *
      variant.ops_per_instruction;
  const double rate = operations / median_ms / 1.0e9;
  const double launch_spread =
      100.0 * (milliseconds.back() - milliseconds.front()) / median_ms;
  std::printf(
      "RESULT,dtype=%s,blocks_per_cu=%d,lds=%zu,median_ms=%.6f,rate=%.6f,"
      "launch_spread_pct=%.4f,samples=%d\n",
      variant.name, requested.blocks_per_cu, requested.dynamic_lds, median_ms,
      rate, launch_spread, kTimedLaunches);
}

int main(int argc, char** argv) {
  bool reverse = false;
  if (argc == 2 && std::strcmp(argv[1], "--reverse") == 0) {
    reverse = true;
  } else if (argc != 1) {
    std::fprintf(stderr, "usage: %s [--reverse]\n", argv[0]);
    return 2;
  }

  int device = 0;
  int cu = 0;
  hipDeviceProp_t props{};
  HIP_OK(hipGetDevice(&device));
  HIP_OK(hipGetDeviceProperties(&props, device));
  HIP_OK(hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount,
                               device));
  if (std::strncmp(props.gcnArchName, "gfx1201", 7) != 0) {
    std::fprintf(stderr, "expected gfx1201, found %s\n", props.gcnArchName);
    return 2;
  }
  std::printf("DEVICE,name=%s,arch=%s,cu=%d,order=%s\n", props.name,
              props.gcnArchName, cu, reverse ? "reverse" : "forward");

  unsigned long long* out = nullptr;
  HIP_OK(hipMalloc(&out, sizeof(*out)));
  HIP_OK(hipMemset(out, 0, sizeof(*out)));

  const Variant forward[] = {{"f16", peak_f16, 2 * 16 * 16 * 16},
                             {"iu8", peak_iu8, 2 * 16 * 16 * 16},
                             {"fp8", peak_fp8, 2 * 16 * 16 * 16},
                             {"iu4", peak_iu4, 2 * 16 * 16 * 32}};
  const Variant backward[] = {{"iu4", peak_iu4, 2 * 16 * 16 * 32},
                              {"fp8", peak_fp8, 2 * 16 * 16 * 16},
                              {"iu8", peak_iu8, 2 * 16 * 16 * 16},
                              {"f16", peak_f16, 2 * 16 * 16 * 16}};
  const Occupancy occupancies[] = {{1, 65536}, {3, 21845}, {8, 0}};
  const Variant* variants = reverse ? backward : forward;

  // One full untimed matrix pays launch/JIT costs and raises the device clock
  // before any reported sample.
  for (const Occupancy& occupancy : occupancies) {
    for (int dtype = 0; dtype < 4; ++dtype) {
      HIP_OK(hipFuncSetAttribute(
          (const void*)variants[dtype].kernel,
          hipFuncAttributeMaxDynamicSharedMemorySize,
          (int)occupancy.dynamic_lds));
      launch(variants[dtype].kernel, cu * occupancy.blocks_per_cu,
             occupancy.dynamic_lds, out);
    }
  }
  HIP_OK(hipDeviceSynchronize());

  for (const Occupancy& occupancy : occupancies) {
    for (int dtype = 0; dtype < 4; ++dtype) {
      run_cell(variants[dtype], occupancy, cu, out);
    }
  }

  HIP_OK(hipFree(out));
  return 0;
}
