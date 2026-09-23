#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#define HIPFIRE_KV_FP8_E4M3 1
#include "../../../kernels/src/kv_cache_write_q8_0_batched.hip"
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP fail %d: %s\n", __LINE__, hipGetErrorString(e)); return 1; } } while (0)
static uint64_t s = 0x12345678u;
static uint64_t nx() {
    s += 0x9e3779b97f4a7c15ull; uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
int main() {
    CK(hipSetDevice(0));
    const int NKV = 4, HD = 256, B = 16;
    const size_t ROW = (size_t)NKV * (HD + 2);
    const size_t CODES = (size_t)NKV * HD;
    const int CAP = 16;
    unsigned char *vn;
    CK(hipMalloc(&vn, CAP * ROW));
    CK(hipMemset(vn, 0, CAP * ROW));
    std::vector<float> hv(B * NKV * HD);
    for (auto& x : hv) {
        double u1 = (double)nx() / (double)UINT64_MAX;
        if (u1 < 1e-12) u1 = 1e-12;
        double u2 = (double)nx() / (double)UINT64_MAX;
        x = (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)) * 0.7f;
    }
    std::vector<int32_t> pos(B);
    for (int i = 0; i < B; i++) pos[i] = i;
    float* dv; int32_t* dp;
    CK(hipMalloc(&dv, hv.size() * 4));
    CK(hipMalloc(&dp, B * 4));
    CK(hipMemcpy(dv, hv.data(), hv.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dp, pos.data(), B * 4, hipMemcpyHostToDevice));
    // single block: head=0,bid=0 only
    {
        dim3 grid(NKV, B, 1), block(32, 1, 1);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_v_batched, grid, block, 0, 0,
            vn, dv, dp, NKV, HD, B);
        CK(hipDeviceSynchronize());
    }
    std::vector<unsigned char> hn(CAP * ROW);
    CK(hipMemcpy(hn.data(), vn, CAP * ROW, hipMemcpyDeviceToHost));
    float amax = 0;
    for (int d = 0; d < HD; d++) {
        float v = hv[d];
        if (fabsf(v) > amax) amax = fabsf(v);
    }
    uint16_t got = hn[CODES] | ((uint16_t)hn[CODES + 1] << 8);
    printf("host amax=%f devscale=%04x\n", amax, got);
    return 0;
}
