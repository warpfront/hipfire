// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
#include "harness.hpp"
// Node harness for the BF16-weight small-M GEMMs (Raon MoE router M128 K2560 F32 x, mHC selector M17 K10240
// BF16 x): every arm is bit-compared with the one-wave-per-output kernel at each row count, then timed with
// events around back-to-back launches (X rotated over >= 2 buffers). Build (per arch):
//   hipcc -O3 --offload-arch=gfx1201 -std=c++17 -o lab benchmarks/small-rows-node/lab.cpp
//   ./lab [--geo=router|selector_bf16x|...] [--rows=1,2,4,...] [--rounds=3] [--bitonly]
// Arms: ref, inc (the previous selection), sr_p1/sr_p4/sr_p8 (forced profiles), sr_sel (the dispatch selection).
// Evidence: receipts/selectors/evidence/small_rows_node_gfx1201.txt. GPU runs go through the fabric.
#include "../../kernels/src/gemm_bf16_xf32_batched.hip"
#include "../../kernels/src/gemm_bf16_xf32_tiled_rows.hip"
#include "../../kernels/src/gemm_bf16_xbf16_rows.hip"
#if 1
#include "cand.hip"
#endif

int main(int argc, char** argv) {
  const char* only = nullptr; int rounds = 3; const char* geosel = nullptr; const char* rowsarg = nullptr; bool bitonly = false;
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--arm=", 6)) only = argv[i] + 6;
    else if (!strncmp(argv[i], "--rounds=", 9)) rounds = atoi(argv[i] + 9);
    else if (!strncmp(argv[i], "--geo=", 6)) geosel = argv[i] + 6;
    else if (!strcmp(argv[i], "--bitonly")) bitonly = true;
    else if (!strncmp(argv[i], "--rows=", 7)) rowsarg = argv[i] + 7;
  }
  hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0));
  const bool g1201 = strstr(pr.gcnArchName, "gfx1201") != nullptr;
  std::vector<Geo> geos = {{"router", 128, 2560, false}, {"selector_bf16x", 17, 10240, true}, {"selector_f32x", 17, 10240, false},
    {"odd_router", 200, 3072, false}, {"odd_sel", 19, 5120, true}, {"odd_small", 3, 512, false}, {"odd_small_bf", 5, 1024, true}, {"m32_bf", 32, 1536, true}, {"m33_f32", 33, 2048, false}};
  if (geosel) { std::vector<Geo> g2; for (auto& g : geos) if (!strcmp(g.name, geosel)) g2.push_back(g); geos = g2; }
  std::vector<int> rowsl = {1, 2, 4, 8, 16, 64, 512, 4096};
  if (rowsarg) { rowsl.clear(); for (char* t = strtok(strdup(rowsarg), ","); t; t = strtok(nullptr, ",")) rowsl.push_back(atoi(t)); }
  std::vector<Arm> arms;
  // per-geometry arms are selected by closure on M/K/xbf16 at launch time
  auto is_bf = [](int M, int K) { return M == 17 && false; };
  (void)is_bf;
  // Reference: one wave per output (all N).
  arms.push_back({"ref", [](const uint16_t* W, const void* X, float* Y, int M, int K, int N) {
    // dispatch on x type by K-size trick is wrong; use global flag
    extern bool g_xbf16;
    if (g_xbf16) hipLaunchKernelGGL(gemm_bf16_xbf16_batched, dim3(M, N), dim3(32), 0, 0, W, (const unsigned short*)X, Y, M, K, N);
    else hipLaunchKernelGGL(gemm_bf16_xf32_batched, dim3(M, N), dim3(32), 0, 0, W, (const float*)X, Y, M, K, N);
  }, 1, 1 << 30});
  arms.push_back({"inc", [g1201](const uint16_t* W, const void* X, float* Y, int M, int K, int N) {
    extern bool g_xbf16;
    const bool tiled = g1201 && N >= 512;
    if (!tiled) { if (g_xbf16) hipLaunchKernelGGL(gemm_bf16_xbf16_batched, dim3(M, N), dim3(32), 0, 0, W, (const unsigned short*)X, Y, M, K, N);
                  else hipLaunchKernelGGL(gemm_bf16_xf32_batched, dim3(M, N), dim3(32), 0, 0, W, (const float*)X, Y, M, K, N); return; }
    const int gy = (N + 3) / 4;
    if (M <= 32) { if (g_xbf16) hipLaunchKernelGGL(gemm_bf16_xbf16_tiled_rows_mp4, dim3((M + 31) / 32, gy), dim3(256), 0, 0, W, (const unsigned short*)X, Y, M, K, N);
                   else hipLaunchKernelGGL(gemm_bf16_xf32_tiled_rows_mp4, dim3((M + 31) / 32, gy), dim3(256), 0, 0, W, (const float*)X, Y, M, K, N); }
    else { if (g_xbf16) hipLaunchKernelGGL(gemm_bf16_xbf16_tiled_rows, dim3((M + 127) / 128, gy), dim3(256), 0, 0, W, (const unsigned short*)X, Y, M, K, N);
           else hipLaunchKernelGGL(gemm_bf16_xf32_tiled_rows, dim3((M + 127) / 128, gy), dim3(256), 0, 0, W, (const float*)X, Y, M, K, N); }
  }, 1, 1 << 30});
#ifdef CAND_ARMS
  CAND_ARMS(arms, g1201)
#endif
  int rc = 0;
  extern bool g_xbf16;
  for (auto& g : geos) { g_xbf16 = g.xbf16; std::vector<Geo> one = {g}; rc |= run_all(one, rowsl, arms, only, bitonly ? 0 : rounds); }
  return rc;
}
bool g_xbf16 = false;
