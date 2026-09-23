#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using i2 = int __attribute__((ext_vector_type(2)));
using i8 = int __attribute__((ext_vector_type(8)));
using f8 = float __attribute__((ext_vector_type(8)));
using u2 = unsigned __attribute__((ext_vector_type(2)));

#define HIP_OK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
  std::exit(1); } } while (0)

static constexpr int LDS_BYTES = 20480;
static constexpr int W0 = 0;
static constexpr int W1 = 4096;
static constexpr int A0 = 8192;
static constexpr int A1 = 12288;
static constexpr int META0 = 16384;
static constexpr int META1 = 18432;

__device__ __forceinline__ void drain_register(i8 (&c)[8], const i2 &wr, const i2 &ar) {
#pragma unroll
  for (int sb = 0; sb < 2; ++sb) {
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) {
        const int q = rg * 4 + nb;
        c[q] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
            true, wr, true, ar, c[q], false);
      }
    }
  }
}

__device__ __forceinline__ void drain_w_lds(i8 (&c)[8], const char *wlds,
                                             const i2 &ar, int wave, int lane) {
#pragma unroll
  for (int sb = 0; sb < 2; ++sb) {
    i2 wr[2];
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      const int g = (wave >> 1) * 2 + rg;
      wr[rg] = *reinterpret_cast<const i2 *>(wlds + (g * 2 + sb) * 256 + lane * 8);
    }
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) {
        const int q = rg * 4 + nb;
        c[q] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
            true, wr[rg], true, ar, c[q], false);
      }
    }
  }
}

__device__ __forceinline__ void drain_wa_lds(i8 (&c)[8], const char *wlds,
                                              const char *alds, int wave, int lane) {
#pragma unroll
  for (int sb = 0; sb < 2; ++sb) {
    i2 wr[2];
    i2 ar[4];
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      const int g = (wave >> 1) * 2 + rg;
      wr[rg] = *reinterpret_cast<const i2 *>(wlds + (g * 2 + sb) * 256 + lane * 8);
    }
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
      const int g = (wave & 1) * 4 + nb;
      ar[nb] = *reinterpret_cast<const i2 *>(alds + (g * 2 + sb) * 256 + lane * 8);
    }
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) {
        const int q = rg * 4 + nb;
        c[q] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
            true, wr[rg], true, ar[nb], c[q], false);
      }
    }
  }
}

__device__ __forceinline__ float2 fold_pair(
    float sum0, float sum1, float sc0, float sc1, float d, int c0, int c1) {
  const float t0 = __fmul_rn(sc0, d);
  const float t1 = __fmul_rn(sc1, d);
  const float p0 = __fmul_rn(t0, (float)c0);
  const float p1 = __fmul_rn(t1, (float)c1);
  return float2{__fadd_rn(sum0, p0), __fadd_rn(sum1, p1)};
}

template <int STEP>
__global__ __launch_bounds__(256) void iu4_ladder(
    const u2 *__restrict__ wg, const u2 *__restrict__ ag,
    const unsigned *__restrict__ mg, float *__restrict__ output,
    unsigned long long *__restrict__ sink, int k128, int gx, int gy) {
  extern __shared__ char lds[];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int block_linear = by * gx + bx;
  const int k64_count = k128 * 2;
  const i2 wr = {(int)(0x11111111u | tid), (int)(0x22222222u | tid)};
  const i2 ar = {(int)(0x33333333u | tid), (int)(0x44444444u | tid)};
  const i8 zi = {0, 0, 0, 0, 0, 0, 0, 0};
  const f8 zf = {0, 0, 0, 0, 0, 0, 0, 0};
  i8 c[8];
  f8 sum[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) { c[q] = zi; sum[q] = zf; }

  if constexpr (STEP == 1) {
    reinterpret_cast<u2 *>(lds + W0)[tid * 2] = u2{(unsigned)wr[0], (unsigned)wr[1]};
    reinterpret_cast<u2 *>(lds + W0)[tid * 2 + 1] = u2{(unsigned)wr[0], (unsigned)wr[1]};
    __syncthreads();
  }

  if constexpr (STEP >= 2) {
    const size_t wi = ((size_t)bx * k64_count) * 512 + tid * 2;
    reinterpret_cast<u2 *>(lds + W0)[tid * 2] = wg[wi];
    reinterpret_cast<u2 *>(lds + W0)[tid * 2 + 1] = wg[wi + 1];
    if constexpr (STEP >= 3) {
      const size_t ai = ((size_t)by * k64_count) * 512 + tid * 2;
      reinterpret_cast<u2 *>(lds + A0)[tid * 2] = ag[ai];
      reinterpret_cast<u2 *>(lds + A0)[tid * 2 + 1] = ag[ai + 1];
    }
    if constexpr (STEP >= 4) {
      const size_t rw = ((size_t)bx * k128) * 256 + tid;
      const size_t ra = ((size_t)gx * k128 + (size_t)by * k128) * 256 + tid;
      reinterpret_cast<unsigned *>(lds + META0)[tid] = mg[rw];
      reinterpret_cast<unsigned *>(lds + META0)[256 + tid] = mg[ra];
    }
    __syncthreads();
  }

  if constexpr (STEP == 0) {
    for (int slab = 0; slab < k64_count; ++slab) drain_register(c, wr, ar);
  } else if constexpr (STEP == 1) {
    for (int slab = 0; slab < k64_count; ++slab) drain_w_lds(c, lds + W0, ar, wave, lane);
  } else {
    int slot = 0;
    for (int slab = 0; slab < k64_count; ++slab) {
      const bool has_next = slab + 1 < k64_count;
      u2 wn0{}, wn1{}, an0{}, an1{};
      unsigned mn0 = 0, mn1 = 0;
      if (has_next) {
        const size_t wi = ((size_t)bx * k64_count + slab + 1) * 512 + tid * 2;
        wn0 = wg[wi]; wn1 = wg[wi + 1];
        if constexpr (STEP >= 3) {
          const size_t ai = ((size_t)by * k64_count + slab + 1) * 512 + tid * 2;
          an0 = ag[ai]; an1 = ag[ai + 1];
        }
        if constexpr (STEP >= 4) {
          if ((slab & 1) && slab + 2 < k64_count) {
            const size_t r = slab / 2 + 1;
            const size_t rw = ((size_t)bx * k128 + r) * 256 + tid;
            const size_t ra = ((size_t)gx * k128 + (size_t)by * k128 + r) * 256 + tid;
            mn0 = mg[rw]; mn1 = mg[ra];
          }
        }
      }
      const char *wcur = lds + (slot ? W1 : W0);
      if constexpr (STEP >= 3) {
        const char *acur = lds + (slot ? A1 : A0);
        drain_wa_lds(c, wcur, acur, wave, lane);
      } else {
        drain_w_lds(c, wcur, ar, wave, lane);
      }

      if constexpr (STEP >= 4) {
        if (slab & 1) {
          volatile const float *meta = reinterpret_cast<volatile const float *>(
              lds + (((slab / 2) & 1) ? META1 : META0));
          const int k_group = lane >> 4;
          const int m_lane = lane & 15;
#pragma unroll
          for (int rg = 0; rg < 2; ++rg) {
            float sc[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const int row = (wave >> 1) * 32 + rg * 16 + 8 * k_group + j;
              sc[j] = meta[row * 2];
            }
#pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
              const int q = rg * 4 + nb;
              const int token = (wave & 1) * 64 + nb * 16 + m_lane;
              const float d = meta[256 + token * 2];
#pragma unroll
              for (int j = 0; j < 8; j += 2) {
                const float2 p = fold_pair(
                    sum[q][j], sum[q][j + 1], sc[j], sc[j + 1], d,
                    c[q][j], c[q][j + 1]);
                sum[q][j] = p.x;
                sum[q][j + 1] = p.y;
              }
              c[q] = zi;
            }
          }
        }
      }

      if (has_next) {
        const int nslot = slot ^ 1;
        reinterpret_cast<u2 *>(lds + (nslot ? W1 : W0))[tid * 2] = wn0;
        reinterpret_cast<u2 *>(lds + (nslot ? W1 : W0))[tid * 2 + 1] = wn1;
        if constexpr (STEP >= 3) {
          reinterpret_cast<u2 *>(lds + (nslot ? A1 : A0))[tid * 2] = an0;
          reinterpret_cast<u2 *>(lds + (nslot ? A1 : A0))[tid * 2 + 1] = an1;
        }
        if constexpr (STEP >= 4) {
          if ((slab & 1) && slab + 2 < k64_count) {
            const int mslot = ((slab / 2 + 1) & 1);
            reinterpret_cast<unsigned *>(lds + (mslot ? META1 : META0))[tid] = mn0;
            reinterpret_cast<unsigned *>(lds + (mslot ? META1 : META0))[256 + tid] = mn1;
          }
        }
      }
      __syncthreads();
      slot ^= 1;
    }
  }

  if constexpr (STEP >= 5) {
    volatile float *slots = reinterpret_cast<volatile float *>(lds);
    const int m_lane = lane & 15;
    const int k_group = lane >> 4;
    const int store_i = tid & 15;
    const int store_x = (tid >> 4) & 15;
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) {
        const int q = rg * 4 + nb;
#pragma unroll
        for (int j = 0; j < 8; ++j)
          slots[wave * 320 + (8 * k_group + j) * 20 + m_lane] = sum[q][j];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int p = k >> 1;
          const int h = k & 1;
          const int owner = p * 2 + h;
          const float value = slots[owner * 320 + store_i * 20 + store_x];
          const int row = p * 32 + rg * 16 + store_i;
          const int col = h * 64 + nb * 16 + store_x;
          output[(size_t)block_linear * 16384 + col * 128 + row] = value;
        }
        __syncthreads();
      }
    }
  }

  unsigned long long check = 0;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    check += (unsigned)c[q][lane & 7];
    if constexpr (STEP >= 4) check += (unsigned)__float_as_uint(sum[q][lane & 7]);
  }
  sink[(size_t)block_linear * 256 + tid] = check;
}

struct Shape { const char *name; int gx, gy, k128; };

template <int STEP>
static double run_shape(const Shape &s, const u2 *wg, const u2 *ag,
                        const unsigned *mg, float *out, unsigned long long *sink) {
  dim3 grid(s.gx, s.gy), block(256);
  for (int i = 0; i < 5; ++i)
    iu4_ladder<STEP><<<grid, block, LDS_BYTES>>>(wg, ag, mg, out, sink, s.k128, s.gx, s.gy);
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> samples;
  for (int i = 0; i < 20; ++i) {
    hipEvent_t a, b; HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
    HIP_OK(hipEventRecord(a));
    iu4_ladder<STEP><<<grid, block, LDS_BYTES>>>(wg, ag, mg, out, sink, s.k128, s.gx, s.gy);
    HIP_OK(hipEventRecord(b)); HIP_OK(hipEventSynchronize(b));
    float ms = 0; HIP_OK(hipEventElapsedTime(&ms, a, b)); samples.push_back(ms * 1000.0f);
    HIP_OK(hipEventDestroy(a)); HIP_OK(hipEventDestroy(b));
  }
  std::sort(samples.begin(), samples.end());
  const double med = 0.5 * (samples[9] + samples[10]);
  int occ = 0; HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, iu4_ladder<STEP>, 256, LDS_BYTES));
  const double ops = (double)s.gx * s.gy * 8.0 * s.k128 * 32.0 * 16384.0;
  std::printf("SHAPE,step=%d,name=%s,median_us=%.3f,TOPS=%.3f,occ=%d\n",
              STEP, s.name, med, ops / med / 1.0e6, occ);
  return med;
}

template <int STEP>
static void run_step(const Shape &gate, const Shape &down,
                     const u2 *wg0, const u2 *ag0, const unsigned *mg0, float *out0,
                     unsigned long long *sink0, const u2 *wg1, const u2 *ag1,
                     const unsigned *mg1, float *out1, unsigned long long *sink1) {
  const double t0 = run_shape<STEP>(gate, wg0, ag0, mg0, out0, sink0);
  const double t1 = run_shape<STEP>(down, wg1, ag1, mg1, out1, sink1);
  const double ops0 = (double)gate.gx * gate.gy * 8.0 * gate.k128 * 32.0 * 16384.0;
  const double ops1 = (double)down.gx * down.gy * 8.0 * down.k128 * 32.0 * 16384.0;
  std::printf("AGG,step=%d,median_us=%.3f,TOPS=%.3f\n", STEP, t0 + t1,
              (ops0 + ops1) / (t0 + t1) / 1.0e6);
}

struct Buffers {
  u2 *w = nullptr, *a = nullptr;
  unsigned *m = nullptr;
  float *out = nullptr;
  unsigned long long *sink = nullptr;
};

static Buffers allocate_shape(const Shape &s) {
  Buffers b;
  const size_t wb = (size_t)s.gx * s.k128 * 2 * 512 * sizeof(u2);
  const size_t ab = (size_t)s.gy * s.k128 * 2 * 512 * sizeof(u2);
  const size_t mb = (size_t)(s.gx + s.gy) * s.k128 * 256 * sizeof(unsigned);
  const size_t ob = (size_t)s.gx * s.gy * 256 * 64 * sizeof(float);
  const size_t sb = (size_t)s.gx * s.gy * 256 * sizeof(unsigned long long);
  HIP_OK(hipMalloc(&b.w, wb)); HIP_OK(hipMalloc(&b.a, ab)); HIP_OK(hipMalloc(&b.m, mb));
  HIP_OK(hipMalloc(&b.out, ob)); HIP_OK(hipMalloc(&b.sink, sb));
  HIP_OK(hipMemset(b.w, 0x11, wb)); HIP_OK(hipMemset(b.a, 0x22, ab));
  HIP_OK(hipMemset(b.m, 0x3f, mb)); HIP_OK(hipMemset(b.out, 0, ob)); HIP_OK(hipMemset(b.sink, 0, sb));
  return b;
}

int main() {
  const Shape gate{"gate_up", 272, 64, 40};
  const Shape down{"down", 40, 64, 136};
  Buffers g = allocate_shape(gate), d = allocate_shape(down);
  run_step<0>(gate, down, g.w, g.a, g.m, g.out, g.sink, d.w, d.a, d.m, d.out, d.sink);
  run_step<1>(gate, down, g.w, g.a, g.m, g.out, g.sink, d.w, d.a, d.m, d.out, d.sink);
  run_step<2>(gate, down, g.w, g.a, g.m, g.out, g.sink, d.w, d.a, d.m, d.out, d.sink);
  run_step<3>(gate, down, g.w, g.a, g.m, g.out, g.sink, d.w, d.a, d.m, d.out, d.sink);
  run_step<4>(gate, down, g.w, g.a, g.m, g.out, g.sink, d.w, d.a, d.m, d.out, d.sink);
  run_step<5>(gate, down, g.w, g.a, g.m, g.out, g.sink, d.w, d.a, d.m, d.out, d.sink);
}
