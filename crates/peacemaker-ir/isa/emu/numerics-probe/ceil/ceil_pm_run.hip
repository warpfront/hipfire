// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Device runner of the Builder `pm_ceil_probe_<arch>` kernel (peacemaker-isa kernels/ceil_probe.rs, emitted by
// `cargo run -p peacemaker-isa --release --features lab --example ceil_probe_emit -- --out-dir DIR`): all 2^32 f32 bit patterns in 64 slabs of
// 2^26, each slab written by the PM kernel (v_ceil_f32 through the Builder), then reduced on the device to the per-chunk
// digests of ceil_sweep.hip and to a count of mismatches against hipcc `ceilf`.
//   usage: ceil_pm_run CO SYMBOL REF_DIGEST_BIN OUT_JSON
// REF_DIGEST_BIN is ceil_digest_<arch>_mode3.bin of ceil_sweep.hip (the inline-asm hardware sweep of the same arch).
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(3); } } while (0)

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// grid = 1024 chunks of the slab, block = 256; chunk digest layout identical to ceil_sweep.hip.
__global__ void reduce_slab(const uint32_t* y, uint32_t slab, uint64_t* dig, unsigned long long* mism) {
  __shared__ uint64_t sh[8];
  const uint32_t chunk = blockIdx.x;
  uint64_t d = 0;
  unsigned long long bad = 0;
  for (uint32_t k = 0; k < 256; ++k) {
    const uint32_t local = (chunk << 16) | (k << 8) | threadIdx.x;
    const uint32_t i = (slab << 26) | local;
    const uint32_t v = y[local];
    d ^= mix64(((uint64_t)i << 32) | v);
    if (__float_as_uint(ceilf(__uint_as_float(i))) != v) ++bad;
  }
  for (int o = 16; o > 0; o >>= 1) {
    uint32_t lo = __shfl_xor((uint32_t)d, o), hi = __shfl_xor((uint32_t)(d >> 32), o);
    d ^= ((uint64_t)hi << 32) | lo;
  }
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = d;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t r = 0;
    for (int i = 0; i < 8; ++i) r ^= sh[i];
    dig[(slab << 10) | chunk] = r;
  }
  if (bad) atomicAdd(mism, bad);
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s CO SYMBOL REF_DIGEST_BIN OUT_JSON\n", argv[0]); return 2; }
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0));
  printf("device: %s %s\n", p.name, p.gcnArchName);
  hipModule_t mod;
  hipFunction_t fn;
  CK(hipModuleLoad(&mod, argv[1]));
  CK(hipModuleGetFunction(&fn, mod, argv[2]));
  std::vector<uint64_t> ref(65536);
  FILE* f = fopen(argv[3], "rb");
  if (!f || fread(ref.data(), 8, 65536, f) != 65536) { fprintf(stderr, "bad ref digest file\n"); return 2; }
  fclose(f);
  const uint32_t SLAB = 1u << 26;
  uint32_t* y;
  uint64_t* dig;
  unsigned long long* mism;
  CK(hipMalloc(&y, (size_t)SLAB * 4));
  CK(hipMalloc(&dig, 65536 * 8));
  CK(hipMalloc(&mism, 8));
  CK(hipMemset(mism, 0, 8));
  CK(hipMemset(y, 0xa5, (size_t)SLAB * 4));
  for (uint32_t slab = 0; slab < 64; ++slab) {
    struct { void* out; uint32_t base, n; } args = {y, slab << 26, SLAB};
    size_t sz = sizeof args;
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, SLAB / 256, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    reduce_slab<<<1024, 256>>>(y, slab, dig, mism);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
  }
  std::vector<uint64_t> h(65536);
  unsigned long long bad = 0;
  CK(hipMemcpy(h.data(), dig, 65536 * 8, hipMemcpyDeviceToHost));
  CK(hipMemcpy(&bad, mism, 8, hipMemcpyDeviceToHost));
  unsigned long long digest_mismatch = 0;
  for (int c = 0; c < 65536; ++c) digest_mismatch += h[c] != ref[c];
  f = fopen(argv[4], "w");
  fprintf(f, "{\"arch\":\"%s\",\"device\":\"%s\",\"symbol\":\"%s\",\"inputs\":4294967296,\"pm_vs_hipcc_ceilf_word_mismatches\":%llu,\"pm_vs_inline_asm_sweep_chunk_digest_mismatches\":%llu,\"chunks\":65536}\n",
          p.gcnArchName, p.name, argv[2], bad, digest_mismatch);
  fclose(f);
  printf("PM kernel vs hipcc ceilf: %llu word mismatches over 2^32; vs inline-asm sweep: %llu/65536 chunk digest mismatches\n", bad, digest_mismatch);
  return (bad || digest_mismatch) ? 1 : 0;
}
