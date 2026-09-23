#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};
static_assert(sizeof(block_i4_128) == 72, "bad block size");

extern "C" __global__ void quantize_int4_mmq_ds128(
    const float*, block_i4_128*, int, int);
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_full_set(
    const char*, const block_i4_128*, float*, int, int, int, int);
extern "C" __global__ void old_silu_mul_mq_rotate_awq_i4_gfx12(
    const float*, const float*, const float*, const float*, const float*,
    float*, block_i4_128*, int, int);
extern "C" __global__ void gemm_up_silu_mq4g256v2_iu4_gfx1201(
    const char*, const block_i4_128*, const float*, const float*,
    const float*, const float*, block_i4_128*, int, int, int);

#define HIP_OK(call) do { \
    hipError_t e_ = (call); \
    if (e_ != hipSuccess) { \
        std::fprintf(stderr, "HIP failure %s:%d: %s\n", __FILE__, __LINE__, \
            hipGetErrorString(e_)); \
        std::exit(2); \
    } \
} while (0)

extern "C" __global__ void compare_bytes(
    const unsigned char* a,
    const unsigned char* b,
    unsigned long long n,
    unsigned long long* mismatches,
    unsigned long long* first
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) {
        atomicAdd(mismatches, 1ull);
        atomicMin(first, i);
    }
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void put_half(unsigned char* p, float x) {
    __half h = __float2half(x);
    std::memcpy(p, &h, sizeof(h));
}

// Deterministic MQ4V2 tensors with exact zero rows, constant rows, alternating
// codes, per-group header changes, and high-dynamic-range rows. The actual
// gate/up matrices are nevertheless produced on-device by the real iu4 GEMM;
// no host gate/up matrix is uploaded.
static std::vector<unsigned char> make_weights(
    int rows, int k, uint64_t salt
) {
    const int groups = k / 256;
    std::vector<unsigned char> out((size_t)rows * groups * 136);
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < groups; ++g) {
            unsigned char* p = out.data() + ((size_t)r * groups + g) * 136;
            const int kind = r % 257;
            float sc0;
            float zp0;
            float sc1;
            float zp1;
            if (kind == 0) {
                sc0 = sc1 = zp0 = zp1 = 0.0f;
            } else if (kind == 1) {
                sc0 = sc1 = 0.0f;
                zp0 = (g & 1) ? 0.5f : -0.5f;
                zp1 = -zp0;
            } else {
                sc0 = 0.0078125f * (float)(1 + ((r + 3 * g) & 7));
                sc1 = 0.0078125f * (float)(1 + ((5 * r + g) & 7));
                zp0 = 0.03125f * (float)(((r + g) % 9) - 4);
                zp1 = 0.03125f * (float)(((3 * r + g) % 11) - 5);
                if (kind == 2) {
                    sc0 *= 32.0f;
                    sc1 *= 32.0f;
                }
            }
            put_half(p + 0, sc0);
            put_half(p + 2, zp0);
            put_half(p + 4, sc1);
            put_half(p + 6, zp1);
            for (int b = 0; b < 128; ++b) {
                uint64_t z = mix64(salt ^ ((uint64_t)r << 32)
                    ^ ((uint64_t)g << 16) ^ (uint64_t)b);
                unsigned lo = (unsigned)((z + 3 * b + r) & 15);
                unsigned hi = (unsigned)(((z >> 8) + 5 * b + g) & 15);
                if ((r & 63) == 3) {
                    lo = (b & 1) ? 0 : 15;
                    hi = 15 - lo;
                }
                p[8 + b] = (unsigned char)(lo | (hi << 4));
            }
        }
    }
    return out;
}

static std::vector<float> make_x(int n, int k) {
    std::vector<float> x((size_t)n * k);
    for (int t = 0; t < n; ++t) {
        for (int j = 0; j < k; ++j) {
            float v;
            if (t == 0) {
                v = 0.0f;
            } else if (t == 1) {
                v = ((j & 1) ? -0.5f : 0.5f);
            } else {
                int q = (int)((mix64(((uint64_t)t << 32) ^ (uint64_t)j)
                    >> 56) & 31) - 15;
                v = (float)q * 0.0625f;
                if ((j & 127) == 0) {
                    v = ((t + j / 128) & 1) ? 96.0f : -96.0f;
                }
                if ((t % 31) == 0 && (j % 257) == 0) {
                    v *= 8.0f;
                }
            }
            x[(size_t)t * k + j] = v;
        }
    }
    return x;
}

template <typename T>
static T* alloc_count(size_t n) {
    T* p = nullptr;
    HIP_OK(hipMalloc((void**)&p, n * sizeof(T)));
    return p;
}

static void upload(void* dst, const void* src, size_t bytes) {
    HIP_OK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

struct Shared {
    int I = 17408;
    int K = 5120;
    char* w_gate = nullptr;
    char* w_up = nullptr;
    char* w_down = nullptr;
    float* awq = nullptr;
    float* s1 = nullptr;
    float* s2 = nullptr;
};

static Shared make_shared() {
    Shared s;
    auto wg = make_weights(s.I, s.K, 0x475445ULL);
    auto wu = make_weights(s.I, s.K, 0x555000ULL);
    auto wd = make_weights(s.K, s.I, 0x444f574eULL);
    s.w_gate = alloc_count<char>(wg.size());
    s.w_up = alloc_count<char>(wu.size());
    s.w_down = alloc_count<char>(wd.size());
    upload(s.w_gate, wg.data(), wg.size());
    upload(s.w_up, wu.data(), wu.size());
    upload(s.w_down, wd.data(), wd.size());

    std::vector<float> awq(s.I);
    std::vector<float> s1(256), s2(256);
    for (int i = 0; i < s.I; ++i) {
        awq[i] = 0.5f + (float)((i * 17) % 257) / 256.0f;
    }
    for (int i = 0; i < 256; ++i) {
        s1[i] = ((i * 13 + 3) & 1) ? -1.0f : 1.0f;
        s2[i] = ((i * 7 + 1) & 2) ? -1.0f : 1.0f;
    }
    s.awq = alloc_count<float>(awq.size());
    s.s1 = alloc_count<float>(s1.size());
    s.s2 = alloc_count<float>(s2.size());
    upload(s.awq, awq.data(), awq.size() * sizeof(float));
    upload(s.s1, s1.data(), s1.size() * sizeof(float));
    upload(s.s2, s2.data(), s2.size() * sizeof(float));
    return s;
}

struct Case {
    int n;
    float* x = nullptr;
    block_i4_128* xin = nullptr;
    float* gate = nullptr;
    float* up = nullptr;
    block_i4_128* old_down = nullptr;
    block_i4_128* new_down = nullptr;
    float* old_y = nullptr;
    float* new_y = nullptr;
};

static Case make_case(const Shared& s, int n) {
    Case c;
    c.n = n;
    auto x = make_x(n, s.K);
    c.x = alloc_count<float>((size_t)n * s.K);
    upload(c.x, x.data(), x.size() * sizeof(float));
    c.xin = alloc_count<block_i4_128>((size_t)(s.K / 128) * n);
    c.gate = alloc_count<float>((size_t)n * s.I);
    c.up = alloc_count<float>((size_t)n * s.I);
    c.old_down = alloc_count<block_i4_128>((size_t)(s.I / 128) * n);
    c.new_down = alloc_count<block_i4_128>((size_t)(s.I / 128) * n);
    c.old_y = alloc_count<float>((size_t)n * s.K);
    c.new_y = alloc_count<float>((size_t)n * s.K);

    dim3 qgrid((s.K + 1023) / 1024, n);
    hipLaunchKernelGGL(quantize_int4_mmq_ds128, qgrid, dim3(256), 0, 0,
        c.x, c.xin, s.K, n);
    HIP_OK(hipGetLastError());
    HIP_OK(hipDeviceSynchronize());
    return c;
}

static void gate(const Shared& s, Case& c) {
    dim3 grid((s.I + 127) / 128, (c.n + 127) / 128);
    hipLaunchKernelGGL(gemm_mq4g256v2_residual_mmq_iu4_full_set,
        grid, dim3(256), 20480, 0,
        s.w_gate, c.xin, c.gate, s.I, s.K, c.n, 0);
    HIP_OK(hipGetLastError());
}

static void old_tail(const Shared& s, Case& c) {
    dim3 up_grid((s.I + 127) / 128, (c.n + 127) / 128);
    hipLaunchKernelGGL(gemm_mq4g256v2_residual_mmq_iu4_full_set,
        up_grid, dim3(256), 20480, 0,
        s.w_up, c.xin, c.up, s.I, s.K, c.n, 0);
    hipLaunchKernelGGL(old_silu_mul_mq_rotate_awq_i4_gfx12,
        dim3(s.I / 256, c.n), dim3(32), 0, 0,
        c.gate, c.up, s.awq, s.s1, s.s2, (float*)nullptr,
        c.old_down, s.I, c.n);
    dim3 down_grid((s.K + 127) / 128, (c.n + 127) / 128);
    hipLaunchKernelGGL(gemm_mq4g256v2_residual_mmq_iu4_full_set,
        down_grid, dim3(256), 20480, 0,
        s.w_down, c.old_down, c.old_y, s.K, s.I, c.n, 0);
    HIP_OK(hipGetLastError());
}

static void new_tail(const Shared& s, Case& c) {
    dim3 up_grid((s.I + 255) / 256, (c.n + 63) / 64);
    hipLaunchKernelGGL(gemm_up_silu_mq4g256v2_iu4_gfx1201,
        up_grid, dim3(256), 25600, 0,
        s.w_up, c.xin, c.gate, s.awq, s.s1, s.s2,
        c.new_down, s.I, s.K, c.n);
    dim3 down_grid((s.K + 127) / 128, (c.n + 127) / 128);
    hipLaunchKernelGGL(gemm_mq4g256v2_residual_mmq_iu4_full_set,
        down_grid, dim3(256), 20480, 0,
        s.w_down, c.new_down, c.new_y, s.K, s.I, c.n, 0);
    HIP_OK(hipGetLastError());
}

static bool compare_down(const Shared& s, Case& c) {
    const unsigned long long bytes =
        (unsigned long long)(s.I / 128) * c.n * sizeof(block_i4_128);
    unsigned long long* count = alloc_count<unsigned long long>(1);
    unsigned long long* first = alloc_count<unsigned long long>(1);
    unsigned long long z = 0;
    unsigned long long none = ~0ull;
    upload(count, &z, sizeof(z));
    upload(first, &none, sizeof(none));
    unsigned blocks = (unsigned)((bytes + 255) / 256);
    hipLaunchKernelGGL(compare_bytes, dim3(blocks), dim3(256), 0, 0,
        (const unsigned char*)c.old_down,
        (const unsigned char*)c.new_down,
        bytes, count, first);
    HIP_OK(hipDeviceSynchronize());
    unsigned long long hc = 0, hf = 0;
    HIP_OK(hipMemcpy(&hc, count, sizeof(hc), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(&hf, first, sizeof(hf), hipMemcpyDeviceToHost));
    std::printf("BYTE_IDENTITY,N=%d,bytes=%llu,mismatches=%llu,first=%s%llu\n",
        c.n, bytes, hc, hf == ~0ull ? "none" : "", hf == ~0ull ? 0ull : hf);
    HIP_OK(hipFree(count));
    HIP_OK(hipFree(first));
    return hc == 0;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return 0.5f * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

static bool time_chains(const Shared& s, Case& c) {
    for (int i = 0; i < 20; ++i) {
        gate(s, c); old_tail(s, c);
        gate(s, c); new_tail(s, c);
    }
    HIP_OK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_OK(hipEventCreate(&start));
    HIP_OK(hipEventCreate(&stop));
    std::vector<float> old_us, new_us;
    for (int i = 0; i < 20; ++i) {
        HIP_OK(hipEventRecord(start));
        gate(s, c); old_tail(s, c);
        HIP_OK(hipEventRecord(stop));
        HIP_OK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_OK(hipEventElapsedTime(&ms, start, stop));
        old_us.push_back(ms * 1000.0f);

        HIP_OK(hipEventRecord(start));
        gate(s, c); new_tail(s, c);
        HIP_OK(hipEventRecord(stop));
        HIP_OK(hipEventSynchronize(stop));
        HIP_OK(hipEventElapsedTime(&ms, start, stop));
        new_us.push_back(ms * 1000.0f);
        std::printf("EVENT,%d,old_us=%.3f,new_us=%.3f\n",
            i, old_us.back(), new_us.back());
    }
    float old_med = median(old_us);
    float new_med = median(new_us);
    float layer_net = (old_med - new_med) / (float)c.n;
    float network_net = layer_net * 64.0f;
    bool ship = network_net >= 6.4f;
    std::printf("CHAIN,N=%d,warmups=20,events=20,old_median_us=%.3f,new_median_us=%.3f,layer_net_us_per_token=%.6f,network64_net_us_per_token=%.6f,ship_threshold=6.4,pass=%d\n",
        c.n, old_med, new_med, layer_net, network_net, ship ? 1 : 0);
    HIP_OK(hipEventDestroy(start));
    HIP_OK(hipEventDestroy(stop));
    return ship;
}

static void free_case(Case& c) {
    HIP_OK(hipFree(c.x));
    HIP_OK(hipFree(c.xin));
    HIP_OK(hipFree(c.gate));
    HIP_OK(hipFree(c.up));
    HIP_OK(hipFree(c.old_down));
    HIP_OK(hipFree(c.new_down));
    HIP_OK(hipFree(c.old_y));
    HIP_OK(hipFree(c.new_y));
}

int main() {
    int count = 0;
    HIP_OK(hipGetDeviceCount(&count));
    if (count != 1) {
        std::fprintf(stderr, "expected exactly one visible ordinal, got %d\n", count);
        return 2;
    }
    hipDeviceProp_t prop{};
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    std::printf("DEVICE,name=%s,arch=%s,cu=%d\n",
        prop.name, prop.gcnArchName, prop.multiProcessorCount);
    if (std::strncmp(prop.gcnArchName, "gfx1201", 7) != 0) {
        std::fprintf(stderr, "requires exact gfx1201\n");
        return 2;
    }

    hipFuncAttributes attr{};
    HIP_OK(hipFuncGetAttributes(&attr,
        (const void*)gemm_up_silu_mq4g256v2_iu4_gfx1201));
    int active = 0;
    HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        gemm_up_silu_mq4g256v2_iu4_gfx1201, 256, 25600));
    std::printf("RESOURCE_RUNTIME,numRegs=%d,staticShared=%zu,localBytes=%zu,maxThreads=%d,dynamicLds=25600,blocksPerMP=%d\n",
        attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes,
        attr.maxThreadsPerBlock, active);

    Shared s = make_shared();
    bool exact = true;
    bool performance = false;
    for (int n : {128, 4096}) {
        Case c = make_case(s, n);
        HIP_OK(hipMemset(c.old_down, 0xA5,
            (size_t)(s.I / 128) * n * sizeof(block_i4_128)));
        HIP_OK(hipMemset(c.new_down, 0x5A,
            (size_t)(s.I / 128) * n * sizeof(block_i4_128)));
        gate(s, c);
        old_tail(s, c);
        new_tail(s, c);
        HIP_OK(hipDeviceSynchronize());
        exact &= compare_down(s, c);
        if (n == 4096 && exact) {
            performance = time_chains(s, c);
        }
        free_case(c);
    }

    HIP_OK(hipFree(s.w_gate));
    HIP_OK(hipFree(s.w_up));
    HIP_OK(hipFree(s.w_down));
    HIP_OK(hipFree(s.awq));
    HIP_OK(hipFree(s.s1));
    HIP_OK(hipFree(s.s2));
    const char* verdict = !exact ? "KILL_BYTE_MISMATCH"
        : !performance ? "KILL_PERFORMANCE" : "PASS";
    std::printf("VERDICT,%s\n", verdict);
    return exact && performance ? 0 : 1;
}
