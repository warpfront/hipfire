// Debug: constant-input writer comparison.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#define HIPFIRE_KV_FP8_E4M3 1
#include "../../../kernels/src/kv_cache_write_q8_0.hip"
#include "../../../kernels/src/kv_cache_write_q8_0_batched.hip"
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP fail %d: %s\n", __LINE__, hipGetErrorString(e)); return 1; } } while (0)
int main() {
    CK(hipSetDevice(0));
    const int NKV = 4, HD = 256;
    const size_t ROW = (size_t)NKV * (HD + 2);
    const int CAP = 16;
    unsigned char *vo, *vn;
    CK(hipMalloc(&vo, CAP * ROW));
    CK(hipMalloc(&vn, CAP * ROW));
    CK(hipMemset(vo, 0, CAP * ROW));
    CK(hipMemset(vn, 0, CAP * ROW));
    // constant 0.5 input, one token
    std::vector<float> hv(NKV * HD, 0.5f);
    float* dv;
    int32_t* dp;
    CK(hipMalloc(&dv, hv.size() * 4));
    CK(hipMalloc(&dp, 4));
    CK(hipMemcpy(dv, hv.data(), hv.size() * 4, hipMemcpyHostToDevice));
    int32_t zero = 0;
    CK(hipMemcpy(dp, &zero, 4, hipMemcpyHostToDevice));
    {
        dim3 grid(NKV, 1, 1), block(32, 1, 1);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3, grid, block, 0, 0, vo, dv, dp, NKV, HD);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_v, grid, block, 0, 0, vn, dv, dp, NKV, HD);
        CK(hipDeviceSynchronize());
    }
    std::vector<unsigned char> ho(CAP * ROW), hn(CAP * ROW);
    CK(hipMemcpy(ho.data(), vo, CAP * ROW, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hn.data(), vn, CAP * ROW, hipMemcpyDeviceToHost));
    printf("old row0 head0 dims0-15:");
    for (int d = 0; d < 16; d++) printf(" %u", ho[d]);
    printf("\nnew grp0 head0 dims0-15:");
    for (int d = 0; d < 16; d++) printf(" %u", hn[(size_t)d * 8]);
    printf("\nold scale h0: %02x %02x  new scale h0: %02x %02x\n",
        ho[1024], ho[1025], hn[1024], hn[1025]);
    printf("old row1 head0 dim0: %u  new grp0 head0 dim0 key1: %u\n", ho[ROW], hn[1]);
    return 0;
}
