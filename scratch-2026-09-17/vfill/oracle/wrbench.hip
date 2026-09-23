#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#define HIPFIRE_KV_FP8_E4M3 1
#include "../../../kernels/src/kv_cache_write_q8_0.hip"
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP fail %d: %s\n", __LINE__, hipGetErrorString(e)); return 1; } } while (0)
int main() {
    CK(hipSetDevice(0));
    const int NKV = 4, HD = 256, CAP = 4096;
    const size_t ROW = (size_t)NKV * (HD + 2);
    std::vector<float> v(NKV * HD, 0.5f);
    float* dv; int32_t* dp;
    unsigned char *c1, *c2;
    CK(hipMalloc(&dv, v.size() * 4)); CK(hipMalloc(&dp, 4));
    CK(hipMalloc(&c1, CAP * ROW)); CK(hipMalloc(&c2, CAP * ROW));
    CK(hipMemcpy(dv, v.data(), v.size() * 4, hipMemcpyHostToDevice));
    int32_t p = 100;
    CK(hipMemcpy(dp, &p, 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const int IT = 2000;
    dim3 g(NKV, 1, 1), b(32, 1, 1);
    float m1, m2;
    CK(hipEventRecord(e0));
    for (int i = 0; i < IT; i++)
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3, g, b, 0, 0, c1, dv, dp, NKV, HD);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&m1, e0, e1));
    CK(hipEventRecord(e0));
    for (int i = 0; i < IT; i++)
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_v, g, b, 0, 0, c2, dv, dp, NKV, HD);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&m2, e0, e1));
    printf("K-writer %.3f us | V-writer %.3f us\n", m1 / IT * 1000.0, m2 / IT * 1000.0);
    // swapped order
    CK(hipEventRecord(e0));
    for (int i = 0; i < IT; i++)
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_v, g, b, 0, 0, c2, dv, dp, NKV, HD);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&m2, e0, e1));
    CK(hipEventRecord(e0));
    for (int i = 0; i < IT; i++)
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3, g, b, 0, 0, c1, dv, dp, NKV, HD);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&m1, e0, e1));
    printf("V-writer %.3f us | K-writer %.3f us (swapped)\n", m2 / IT * 1000.0, m1 / IT * 1000.0);
    return 0;
}
