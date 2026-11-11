// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Forced-profile and selection arms for the small-rows kernels (see lab.cpp).
#include "../../kernels/src/gemm_bf16_small_rows.hip"
extern bool g_xbf16;
// forced profile arms + the dispatch selection (sr_sel mirrors the shipped SMALL_ROWS_*_MIN_ROWS of stream_mix.rs: router
// 16/256, selector 80/512; the evidence file's j5919 sr_sel rows were taken with the earlier 9/256 and 33/512 and the
// shipped thresholds come from its sr_p1/sr_p4/sr_p8 rows plus the j5921 threshold rows)
static inline void sr_launch(const uint16_t* W, const void* X, float* Y, int M, int K, int N, int prof) {
  // prof: 1,4,8 -> NT
  if (g_xbf16) {
    const unsigned short* x = (const unsigned short*)X;
    if (prof == 1) hipLaunchKernelGGL(gemm_bf16_xbf16_small_rows_s1, dim3((M + 16) / 17, N), dim3(17 * 32), 0, 0, W, x, Y, M, K, N);
    else if (prof == 4) hipLaunchKernelGGL(gemm_bf16_xbf16_small_rows_s4, dim3((M + 17) / 18, (N + 3) / 4), dim3(6 * 32), 0, 0, W, x, Y, M, K, N);
    else hipLaunchKernelGGL(gemm_bf16_xbf16_small_rows_s8, dim3((M + 19) / 20, (N + 7) / 8), dim3(4 * 32), 0, 0, W, x, Y, M, K, N);
  } else {
    const float* x = (const float*)X;
    if (prof == 1) hipLaunchKernelGGL(gemm_bf16_xf32_small_rows_r1, dim3((M + 7) / 8, N), dim3(8 * 32), 0, 0, W, x, Y, M, K, N);
    else if (prof == 4) hipLaunchKernelGGL(gemm_bf16_xf32_small_rows_r4, dim3((M + 31) / 32, (N + 3) / 4), dim3(8 * 32), 0, 0, W, x, Y, M, K, N);
    else hipLaunchKernelGGL(gemm_bf16_xf32_small_rows_r8, dim3((M + 63) / 64, (N + 7) / 8), dim3(8 * 32), 0, 0, W, x, Y, M, K, N);
  }
}
#define CAND_ARMS(arms, g1201) \
  arms.push_back({"sr_p1", [](const uint16_t* W, const void* X, float* Y, int M, int K, int N) { sr_launch(W, X, Y, M, K, N, 1); }, 1, 1 << 30}); \
  arms.push_back({"sr_p4", [](const uint16_t* W, const void* X, float* Y, int M, int K, int N) { sr_launch(W, X, Y, M, K, N, 4); }, 1, 1 << 30}); \
  arms.push_back({"sr_p8", [](const uint16_t* W, const void* X, float* Y, int M, int K, int N) { sr_launch(W, X, Y, M, K, N, 8); }, 1, 1 << 30}); \
  arms.push_back({"sr_sel", [](const uint16_t* W, const void* X, float* Y, int M, int K, int N) { \
      int prof; if (M <= 32) prof = N < 80 ? 1 : (N < 512 ? 4 : 8); else prof = N < 16 ? 1 : (N < 256 ? 4 : 8); sr_launch(W, X, Y, M, K, N, prof); }, 1, 1 << 30});
