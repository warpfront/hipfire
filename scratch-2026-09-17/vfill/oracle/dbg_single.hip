#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#define HIPFIRE_KV_FP8_E4M3 1
#include "../../../kernels/src/kv_cache_write_q8_0.hip"
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP fail %d: %s\n", __LINE__, hipGetErrorString(e)); return 1; } } while (0)
static uint64_t s = 0x99u;
static uint64_t nx() {
    s += 0x9e3779b97f4a7c15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
int main() {
    CK(hipSetDevice(0));
    const int NKV = 4, HD = 256;
    const size_t ROW = (size_t)NKV * (HD + 2);
    const size_t CODES = (size_t)NKV * HD;
    const int CAP = 16;
    unsigned char *vo, *vn;
    CK(hipMalloc(&vo, CAP * ROW));
    CK(hipMalloc(&vn, CAP * ROW));
    CK(hipMemset(vo, 0, CAP * ROW));
    CK(hipMemset(vn, 0, CAP * ROW));
    std::vector<float> hv(NKV * HD);
    for (auto& x : hv) {
        double u1 = (double)nx() / (double)UINT64_MAX;
        if (u1 < 1e-12) u1 = 1e-12;
        double u2 = (double)nx() / (double)UINT64_MAX;
        x = (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)) * 0.7f;
    }
    float* dv;
    int32_t* dp;
    CK(hipMalloc(&dv, hv.size() * 4));
    CK(hipMalloc(&dp, 4));
    CK(hipMemcpy(dv, hv.data(), hv.size() * 4, hipMemcpyHostToDevice));
    int32_t p3 = 3;
    CK(hipMemcpy(dp, &p3, 4, hipMemcpyHostToDevice));
    {
        dim3 grid(NKV, 1, 1), block(32, 1, 1);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3, grid, block, 0, 0, vo, dv, dp, NKV, HD);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_v, grid, block, 0, 0, vn, dv, dp, NKV, HD);
        CK(hipDeviceSynchronize());
    }
    std::vector<unsigned char> ho(CAP * ROW), hn(CAP * ROW);
    CK(hipMemcpy(ho.data(), vo, CAP * ROW, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hn.data(), vn, CAP * ROW, hipMemcpyDeviceToHost));
    int bad = 0;
    int t = 3;
    for (int h = 0; h < NKV; h++) {
        for (int d = 0; d < HD; d++) {
            unsigned char a = ho[(size_t)t * ROW + (size_t)h * HD + d];
            unsigned char b = hn[(size_t)(t >> 3) * ROW * 8
                + (size_t)h * HD * 8 + (size_t)d * 8 + (size_t)(t & 7)];
            if (a != b && bad++ < 10)
                printf("CODE h=%d d=%d old=%u new=%u\n", h, d, a, b);
        }
        for (int i = 0; i < 2; i++) {
            unsigned char a = ho[(size_t)t * ROW + CODES + (size_t)h * 2 + i];
            unsigned char b = hn[(size_t)t * ROW + CODES + (size_t)h * 2 + i];
            if (a != b && bad++ < 10)
                printf("SCALE h=%d i=%d old=%u new=%u\n", h, i, a, b);
        }
    }
    printf(bad ? "SINGLE MISMATCH %d\n" : "SINGLE MATCH\n", bad);
    return 0;
}
