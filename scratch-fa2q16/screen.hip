#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); exit(2); \
} } while (0)

static constexpr int N = 8192, H = 24, HKV = 4, D = 256, L = 16;
static constexpr size_t Q_ELEMENTS = size_t(N) * H * D;
static constexpr size_t KV_BYTES = size_t(N) * HKV * (D / 32) * 34;

struct Arm { hipModule_t module; hipFunction_t kernel; int block, grid; const char *name; };

static void launch(const Arm &a, _Float16 *q, unsigned char *k, unsigned char *v,
                   float *out, int *positions, int layer) {
    auto *kp = k + size_t(layer) * KV_BYTES;
    auto *vp = v + size_t(layer) * KV_BYTES;
    int nh = H, nkv = HKV, dim = D, batch = N;
    float scale = 1.0f / sqrtf(float(D));
    void *args[] = {&q, &kp, &vp, &out, &positions, &nh, &nkv, &dim, &batch, &scale};
    HIP_OK(hipModuleLaunchKernel(a.kernel, a.grid, HKV, 1, a.block, 1, 1,
                                 32768, nullptr, args, nullptr));
}

static float time_pass(const Arm &a, _Float16 *q, unsigned char *k, unsigned char *v,
                       float *out, int *positions) {
    hipEvent_t start, end;
    HIP_OK(hipEventCreate(&start)); HIP_OK(hipEventCreate(&end));
    HIP_OK(hipEventRecord(start));
    for (int layer = 0; layer < L; ++layer) launch(a, q, k, v, out, positions, layer);
    HIP_OK(hipEventRecord(end)); HIP_OK(hipEventSynchronize(end));
    float ms = 0; HIP_OK(hipEventElapsedTime(&ms, start, end));
    HIP_OK(hipEventDestroy(start)); HIP_OK(hipEventDestroy(end));
    return ms;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: screen incumbent.hsaco candidate.hsaco\n"); return 2; }
    hipDeviceProp_t props;
    HIP_OK(hipGetDeviceProperties(&props, 0));
    printf("device=%s gfx=%s N=%d H=%d HKV=%d D=%d layers=%d KT=32\n",
           props.name, props.gcnArchName, N, H, HKV, D, L);
    Arm arms[2]{};
    for (int i = 0; i < 2; ++i) {
        arms[i].name = i ? "candidate" : "incumbent";
        arms[i].block = i ? 256 : 128;
        arms[i].grid = N / (i ? 16 : 8);
        HIP_OK(hipModuleLoad(&arms[i].module, argv[i + 1]));
        HIP_OK(hipModuleGetFunction(&arms[i].kernel, arms[i].module,
                                    "attention_q8_0_fa2_gqa_gfx11"));
        int resident = 0;
        HIP_OK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, arms[i].kernel, arms[i].block, 32768));
        printf("arm=%s block=%d grid=%d resident_blocks_per_CU=%d\n",
               arms[i].name, arms[i].block, arms[i].grid, resident);
    }
    std::vector<_Float16> hq(Q_ELEMENTS);
    std::vector<int> hp(N);
    // Nonuniform, deterministic Q8 records. Vary both code patterns and scale.
    std::vector<unsigned char> hk(size_t(L) * KV_BYTES), hv(size_t(L) * KV_BYTES);
    for (size_t i = 0; i < hq.size(); ++i)
        hq[i] = (_Float16)((int((i * 1217 + i / 17) % 251) - 125) * 0.0012f);
    for (int i = 0; i < N; ++i) hp[i] = i;
    for (int layer = 0; layer < L; ++layer) {
        for (size_t block = 0; block < KV_BYTES / 34; ++block) {
            size_t off = size_t(layer) * KV_BYTES + block * 34;
            _Float16 ks = (_Float16)(0.0020f + ((block + layer * 3) % 11) * 0.00015f);
            _Float16 vs = (_Float16)(0.0015f + ((block + layer * 7) % 13) * 0.00010f);
            memcpy(&hk[off], &ks, 2); memcpy(&hv[off], &vs, 2);
            for (int c = 0; c < 32; ++c) {
                hk[off + 2 + c] = (unsigned char)(int((block * 7 + c * 19 + layer * 11) % 121) - 60);
                hv[off + 2 + c] = (unsigned char)(int((block * 17 + c * 13 + layer * 23) % 101) - 50);
            }
        }
    }
    _Float16 *q; unsigned char *k, *v; float *out; int *pos;
    HIP_OK(hipMalloc(&q, Q_ELEMENTS * sizeof(_Float16)));
    HIP_OK(hipMalloc(&k, hk.size())); HIP_OK(hipMalloc(&v, hv.size()));
    HIP_OK(hipMalloc(&out, Q_ELEMENTS * sizeof(float))); HIP_OK(hipMalloc(&pos, N * sizeof(int)));
    HIP_OK(hipMemcpy(q, hq.data(), Q_ELEMENTS * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(k, hk.data(), hk.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(v, hv.data(), hv.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(pos, hp.data(), N * sizeof(int), hipMemcpyHostToDevice));
    std::vector<float> reference(Q_ELEMENTS), actual(Q_ELEMENTS);
    // Compare every output element of all sixteen layers, not a sampled head.
    size_t neq = 0, nonfinite = 0;
    float max_abs = 0;
    for (int layer = 0; layer < L; ++layer) {
        launch(arms[0], q, k, v, out, pos, layer);
        HIP_OK(hipMemcpy(reference.data(), out, reference.size() * sizeof(float), hipMemcpyDeviceToHost));
        launch(arms[1], q, k, v, out, pos, layer);
        HIP_OK(hipMemcpy(actual.data(), out, actual.size() * sizeof(float), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < Q_ELEMENTS; ++i) {
            if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) ++nonfinite;
            if (memcmp(&reference[i], &actual[i], sizeof(float))) ++neq;
            max_abs = std::max(max_abs, fabsf(reference[i] - actual[i]));
        }
    }
    printf("oracle compared=%zu unequal=%zu nonfinite=%zu max_abs=%.9g\n",
           size_t(L) * Q_ELEMENTS, neq, nonfinite, max_abs);
    if (nonfinite || max_abs > 0.0005f) return 3;
    for (int arm = 0; arm < 2; ++arm) time_pass(arms[arm], q, k, v, out, pos);
    // ABBA (two independent repetitions) within one process, GPU-event time.
    for (int rep = 0; rep < 2; ++rep) {
        const int order[] = {0, 1, 1, 0};
        for (int slot = 0; slot < 4; ++slot) {
            int arm = order[slot];
            float ms = time_pass(arms[arm], q, k, v, out, pos);
            printf("rep=%d slot=%d arm=%s ms=%.3f\n", rep + 1, slot + 1, arms[arm].name, ms);
            fflush(stdout);
        }
    }
    HIP_OK(hipFree(pos)); HIP_OK(hipFree(out)); HIP_OK(hipFree(v)); HIP_OK(hipFree(k)); HIP_OK(hipFree(q));
    for (auto &arm : arms) HIP_OK(hipModuleUnload(arm.module));
}
