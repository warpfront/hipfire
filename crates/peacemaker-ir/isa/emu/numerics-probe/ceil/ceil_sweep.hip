// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exhaustive 2^32 hardware sweep of `v_ceil_f32` (inline asm, VOP1) for one architecture, under every FP32 MODE
// denormal setting, against
//   (a) hipcc `ceilf` (the HIP anchor), compared on the device, and
//   (b) the peacemaker-ir emulator model, compared on the host through per-chunk digests
//       (`cargo test -p peacemaker-ir --lib ceil_hardware_digest_sweep -- --ignored`, see README in this dir).
//
// Outputs (OUTDIR):
//   ceil_digest_<arch>_mode<M>.bin   65536 little-endian u64: chunk c (inputs c*65536 .. c*65536+65535) digest
//   ceil_control_<arch>_mode<M>.bin  same layout for `v_mul_f32 x, 1.0` (proves the MODE write took effect)
//   ceil_edges_<arch>.tsv            raw hardware results for a curated edge list under each mode
//   ceil_sweep_<arch>.json           device-side comparison counters, hipcc-ceilf vs asm
//
// digest(chunk) = XOR over i of mix64((u64(i) << 32) | y_i) with mix64 = splitmix64 finalizer.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(3); } } while (0)

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

template <int MODE>
__device__ __forceinline__ void set_denorm_mode() {
  // MODE register FP_DENORM single-precision field = bits [5:4]: 0 flush in+out, 1 flush out, 2 flush in, 3 none.
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), %0\n\ts_nop 7" ::"i"(MODE) : "memory");
}

__device__ __forceinline__ uint32_t hw_ceil(uint32_t x) {
  uint32_t y;
  asm volatile("v_ceil_f32 %0, %1" : "=v"(y) : "v"(x));
  return y;
}
__device__ __forceinline__ uint32_t hw_mul1(uint32_t x) {
  uint32_t y;
  asm volatile("v_mul_f32 %0, %1, 1.0" : "=v"(y) : "v"(x));
  return y;
}

__device__ __forceinline__ uint64_t block_xor(uint64_t v, uint64_t* sh) {
  for (int o = 16; o > 0; o >>= 1) {
    uint32_t lo = __shfl_xor((uint32_t)v, o), hi = __shfl_xor((uint32_t)(v >> 32), o);
    v ^= ((uint64_t)hi << 32) | lo;
  }
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) sh[w] = v;
  __syncthreads();
  uint64_t r = 0;
  for (int i = 0; i < (int)(blockDim.x >> 5); ++i) r ^= sh[i];
  __syncthreads();
  return r;
}

// grid = 65536 chunks, block = 256 threads; each thread covers 256 inputs of its chunk.
template <int MODE>
__global__ void sweep(uint64_t* dig, uint64_t* ctl, unsigned long long* mism, uint32_t* first) {
  __shared__ uint64_t sh[32];
  const uint32_t chunk = blockIdx.x;
  uint64_t d = 0, c = 0;
  unsigned long long bad = 0;
  set_denorm_mode<MODE>();
  for (uint32_t k = 0; k < 256; ++k) {
    const uint32_t i = (chunk << 16) | (k << 8) | threadIdx.x;
    const uint32_t y = hw_ceil(i);
    d ^= mix64(((uint64_t)i << 32) | y);
    c ^= mix64(((uint64_t)i << 32) | hw_mul1(i));
    if (MODE == 3) {
      // hipcc anchor in the default (denormals kept) mode: the product compiles ceilf to the native opcode.
      const float f = __uint_as_float(i);
      const uint32_t z = __float_as_uint(ceilf(f));
      if (z != y) {
        ++bad;
        uint32_t slot = atomicAdd(&first[0], 1u);
        if (slot < 64) { first[1 + slot * 3] = i; first[2 + slot * 3] = y; first[3 + slot * 3] = z; }
      }
    }
  }
  d = block_xor(d, sh);
  c = block_xor(c, sh);
  if (threadIdx.x == 0) { dig[chunk] = d; ctl[chunk] = c; }
  if (bad) atomicAdd(mism, bad);
}

template <int MODE>
__global__ void edges(const uint32_t* in, uint32_t* out, uint32_t* outc, int n) {
  const int i = threadIdx.x;
  set_denorm_mode<MODE>();
  if (i < n) { out[i] = hw_ceil(in[i]); outc[i] = hw_mul1(in[i]); }
}

static uint32_t kEdges[] = {
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00400000, 0x80400000, 0x00800000, 0x80800000,
    0x3f800000, 0xbf800000, 0x3f000000, 0xbf000000, 0x3effffff, 0xbeffffff, 0x3f7fffff, 0xbf7fffff, 0x3f800001, 0xbf800001,
    0x40000000, 0xc0000000, 0x3fc00000, 0xbfc00000, 0x40200000, 0xc0200000, 0x4b000000, 0xcb000000, 0x4afffffe, 0xcafffffe,
    0x4affffff, 0xcaffffff, 0x4b000001, 0xcb000001, 0x4b800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
    0x7fc00000, 0xffc00000, 0x7fc00001, 0xffc00001, 0x7fa00000, 0xffa00000, 0x7f800001, 0xff800001, 0x7fbfffff, 0xffbfffff,
    0x7fffffff, 0xffffffff, 0x7fc0beef, 0xffc0beef, 0x7fa0beef, 0xffa0beef, 0x3dcccccd, 0xbdcccccd, 0x33800000, 0xb3800000,
    0x0000ffff, 0x8000ffff, 0x3f7ffffe, 0x3f800002, 0xbf7ffffe, 0xc0400000, 0x40400000, 0x40490fdb, 0xc0490fdb};

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s OUTDIR ARCH\n", argv[0]); return 2; }
  const std::string out = argv[1], arch = argv[2];
  std::string cmd = "mkdir -p " + out;
  if (system(cmd.c_str())) return 2;
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0));
  printf("device: %s %s\n", p.name, p.gcnArchName);
  uint64_t *dig, *ctl;
  unsigned long long* mism;
  uint32_t* first;
  CK(hipMalloc(&dig, 65536 * 8));
  CK(hipMalloc(&ctl, 65536 * 8));
  CK(hipMalloc(&mism, 8));
  CK(hipMalloc(&first, 4 * 200));
  const int ne = sizeof(kEdges) / 4;
  uint32_t *ein, *eout, *eoutc;
  CK(hipMalloc(&ein, 4 * ne));
  CK(hipMalloc(&eout, 4 * ne));
  CK(hipMalloc(&eoutc, 4 * ne));
  CK(hipMemcpy(ein, kEdges, 4 * ne, hipMemcpyHostToDevice));
  std::vector<uint32_t> eres[4], ectl[4];
  unsigned long long host_mism = 0;
  std::vector<uint32_t> hfirst(200);
  for (int mode = 0; mode < 4; ++mode) {
    CK(hipMemset(mism, 0, 8));
    CK(hipMemset(first, 0, 4 * 200));
    switch (mode) {
      case 0: sweep<0><<<65536, 256>>>(dig, ctl, mism, first); edges<0><<<1, 128>>>(ein, eout, eoutc, ne); break;
      case 1: sweep<1><<<65536, 256>>>(dig, ctl, mism, first); edges<1><<<1, 128>>>(ein, eout, eoutc, ne); break;
      case 2: sweep<2><<<65536, 256>>>(dig, ctl, mism, first); edges<2><<<1, 128>>>(ein, eout, eoutc, ne); break;
      case 3: sweep<3><<<65536, 256>>>(dig, ctl, mism, first); edges<3><<<1, 128>>>(ein, eout, eoutc, ne); break;
    }
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    std::vector<uint64_t> h(65536), hc(65536);
    CK(hipMemcpy(h.data(), dig, 65536 * 8, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hc.data(), ctl, 65536 * 8, hipMemcpyDeviceToHost));
    char name[256];
    snprintf(name, sizeof name, "%s/ceil_digest_%s_mode%d.bin", out.c_str(), arch.c_str(), mode);
    FILE* f = fopen(name, "wb"); fwrite(h.data(), 8, 65536, f); fclose(f);
    snprintf(name, sizeof name, "%s/ceil_control_%s_mode%d.bin", out.c_str(), arch.c_str(), mode);
    f = fopen(name, "wb"); fwrite(hc.data(), 8, 65536, f); fclose(f);
    eres[mode].resize(ne); ectl[mode].resize(ne);
    CK(hipMemcpy(eres[mode].data(), eout, 4 * ne, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ectl[mode].data(), eoutc, 4 * ne, hipMemcpyDeviceToHost));
    if (mode == 3) {
      CK(hipMemcpy(&host_mism, mism, 8, hipMemcpyDeviceToHost));
      CK(hipMemcpy(hfirst.data(), first, 4 * 200, hipMemcpyDeviceToHost));
    }
    printf("mode %d done\n", mode);
  }
  char name[256];
  snprintf(name, sizeof name, "%s/ceil_edges_%s.tsv", out.c_str(), arch.c_str());
  FILE* f = fopen(name, "w");
  fprintf(f, "input\tmode0\tmode1\tmode2\tmode3\tmul1_mode0\tmul1_mode1\tmul1_mode2\tmul1_mode3\n");
  for (int i = 0; i < ne; ++i) {
    fprintf(f, "%08x", kEdges[i]);
    for (int m = 0; m < 4; ++m) fprintf(f, "\t%08x", eres[m][i]);
    for (int m = 0; m < 4; ++m) fprintf(f, "\t%08x", ectl[m][i]);
    fprintf(f, "\n");
  }
  fclose(f);
  snprintf(name, sizeof name, "%s/ceil_sweep_%s.json", out.c_str(), arch.c_str());
  f = fopen(name, "w");
  fprintf(f, "{\"arch\":\"%s\",\"device\":\"%s\",\"inputs\":4294967296,\"asm_vs_hipcc_ceilf_mismatches_mode3\":%llu,\"first\":[", p.gcnArchName, p.name, host_mism);
  const uint32_t nfirst = hfirst[0] < 64 ? hfirst[0] : 64;
  for (uint32_t i = 0; i < nfirst; ++i)
    fprintf(f, "%s[\"%08x\",\"%08x\",\"%08x\"]", i ? "," : "", hfirst[1 + i * 3], hfirst[2 + i * 3], hfirst[3 + i * 3]);
  fprintf(f, "]}\n");
  fclose(f);
  printf("asm vs hipcc ceilf mismatches (mode 3, 2^32 inputs): %llu\n", host_mism);
  return 0;
}
