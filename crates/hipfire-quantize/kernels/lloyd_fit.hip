// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// One unconstrained Lloyd iteration over a 1024-element chunk per thread.
// Partial sums are written in thread-index order so the host can combine
// them in a fixed order. The C++-side spike this came from is in
// tools/lloyd_fit_gpu.hip.

#include <hip/hip_runtime.h>

// The CPU reference does separate multiply and add; FMA contraction changes
// the f64 rounding of the reduction.
#pragma clang fp contract(off)

#define LLOYD_CHUNK 1024
#define LLOYD_LANES 34

extern "C" __global__ void lloyd_partial(
    const double* __restrict__ vals,
    const double* __restrict__ wts,
    const double* __restrict__ levels,
    double* __restrict__ partials,
    unsigned long long n)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long start = (unsigned long long)tid * LLOYD_CHUNK;
  if (start >= n) return;
  double L[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) L[k] = levels[k];
  double sw[16] = {0.0}, sv[16] = {0.0};
  double sse = 0.0, wsum = 0.0;
  for (unsigned long long i = start; i < start + LLOYD_CHUNK && i < n; ++i) {
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
  double* p = partials + (size_t)tid * LLOYD_LANES;
#pragma unroll
  for (int k = 0; k < 16; ++k) { p[k] = sw[k]; p[16 + k] = sv[k]; }
  p[32] = sse;
  p[33] = wsum;
}

// Score four candidate level-sets in one element pass. `levels4` is four
// contiguous 16-double sets. Each chunk writes 4 sse + 4 wsum doubles in
// candidate order; the host combines chunks in launch order.
extern "C" __global__ void lloyd_polish_partial(
    const double* __restrict__ vals,
    const double* __restrict__ wts,
    const double* __restrict__ levels4,
    double* __restrict__ partials,
    unsigned long long n)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long start = (unsigned long long)tid * LLOYD_CHUNK;
  if (start >= n) return;
  double L[4][16];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
#pragma unroll
    for (int k = 0; k < 16; ++k) L[c][k] = levels4[c * 16 + k];
  }
  double sse[4] = {0.0, 0.0, 0.0, 0.0};
  double wsum[4] = {0.0, 0.0, 0.0, 0.0};
  for (unsigned long long i = start; i < start + LLOYD_CHUNK && i < n; ++i) {
    const double v = vals[i], wt = wts[i];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      int idx = 15;
#pragma unroll
      for (int k = 0; k < 15; ++k) {
        const double mid = 0.5 * (L[c][k] + L[c][k + 1]);
        if (v < mid) { idx = k; break; }
      }
      const double e = v - L[c][idx];
      sse[c] += wt * e * e;
      wsum[c] += wt;
    }
  }
  double* p = partials + (size_t)tid * 8;
#pragma unroll
  for (int c = 0; c < 4; ++c) { p[c] = sse[c]; p[4 + c] = wsum[c]; }
}
