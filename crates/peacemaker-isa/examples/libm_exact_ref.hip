// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// hipcc reference kernels + harness helpers of examples/libm_exact.rs. Build per arch with the product flags
// (crates/rdna-compute/src/compiler.rs `core_hipcc_flags`):
//   /opt/rocm/bin/hipcc --genco -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6 \
//       --offload-arch=gfx1201 libm_exact_ref.hip -o libm_exact_ref.gfx1201.co
#include <hip/hip_runtime.h>
typedef unsigned int u32;
typedef unsigned long long u64;

extern "C" __global__ void k_sinf(const float* x, float* y, u32 n) { u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = sinf(x[i]); }
extern "C" __global__ void k_cosf(const float* x, float* y, u32 n) { u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = cosf(x[i]); }
extern "C" __global__ void k_powf(const float* x, const float* e, float* y, u32 n) { u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = powf(x[i], e[i]); }

// x[i] = bit pattern base + i
extern "C" __global__ void k_iota(u32* x, u32 base, u32 n) { u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = base + i; }

static __device__ u64 mix(u64 z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static __device__ float bits(u32 b) { return __uint_as_float(b); }
// A float with a uniformly random exponent in [lo, hi] (biased) and a random mantissa; sign random.
static __device__ float rnd_exp(u64 r, int lo, int hi, bool neg_ok) {
  u32 e = lo + (u32)((r >> 8) % (u32)(hi - lo + 1));
  u32 m = (u32)(r >> 32) & 0x7fffff;
  u32 s = neg_ok ? (u32)(r & 1) << 31 : 0;
  return bits(s | (e << 23) | m);
}
// random (x, y) pairs, mixed distributions. class by (r % 100).
extern "C" __global__ void k_gen_pow(u32* xs, u32* es, u64 seed, u32 n) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  u64 r0 = mix(seed ^ ((u64)i << 1)), r1 = mix(seed ^ ((u64)i << 1 | 1)), r2 = mix(r0 ^ 0x1234567ULL);
  u32 c = (u32)((r0 >> 40) % 100);
  float x, y;
  if (c < 20) { x = bits((u32)r1); y = bits((u32)(r1 >> 32)); }                                  // raw bit patterns
  else if (c < 50) { x = rnd_exp(r1, 100, 154, true); y = rnd_exp(r2, 110, 134, true); }        // moderate magnitudes
  else if (c < 60) { x = bits(0x3f800000u + (int)((r1 >> 8) % 4097) - 2048); y = rnd_exp(r2, 120, 150, true); } // x near 1
  else if (c < 70) { x = rnd_exp(r1, 1, 254, false); y = (float)(int)((r2 >> 8) % 4097 - 2048) / 16.0f; }      // positive x, k/16
  else if (c < 80) { x = rnd_exp(r1, 110, 144, true); y = (float)(int)((r2 >> 8) % 129 - 64); }                  // integer y, any sign x
  else if (c < 90) { x = rnd_exp(r1, 0, 2, true); y = rnd_exp(r2, 118, 140, true); }                   // subnormal / tiny x
  else { x = rnd_exp(r1, 1, 254, true); y = (float)(int)((r2 >> 8) % 2001 - 1000) * 0.5f; }              // half-integers
  xs[i] = __float_as_uint(x); es[i] = __float_as_uint(y);
}

// mismatch count + the first 8 (index, ref, got) triples. cnt[0] = mismatches, cnt[1 + 3k ...].
extern "C" __global__ void k_cmp(const u32* a, const u32* b, u32 n, u32* cnt) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  u32 va = a[i], vb = b[i];
  if (va != vb) { u32 k = atomicAdd(&cnt[0], 1u); if (k < 8) { cnt[1 + 3 * k] = i; cnt[2 + 3 * k] = va; cnt[3 + 3 * k] = vb; } }
}

// self-test control: flip bit 0 of p[i] (the harness must then report exactly one mismatch).
extern "C" __global__ void k_flip(u32* p, u32 i) { if (blockIdx.x == 0 && threadIdx.x == 0) p[i] ^= 1u; }
