#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#define HIPFIRE_KV_FP8_E4M3 1
#include "../../../kernels/src/kv_cache_write_q8_0_batched.hip"
#include "o_kv.hip"
#include "../../../kernels/src/attention_q8_0_kv.hip"
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP fail %d: %s\n", __LINE__, hipGetErrorString(e)); return 1; } } while (0)
static uint64_t s = 7u;
static uint64_t nx() {
    s += 0x9e3779b97f4a7c15ull; uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
int main() {
    CK(hipSetDevice(0));
    const int NH = 24, NKV = 4, HD = 256, SEQ = 1024, CAP = 1024;
    const size_t ROW = (size_t)NKV * (HD + 2);
    const float sc = 1.0f / sqrtf((float)HD);
    std::vector<float> q(NH * HD), k(SEQ * NKV * HD), v(SEQ * NKV * HD);
    for (auto& x : q) x = (float)(nx() % 2000) / 1000.0f - 1.0f;
    for (auto& x : k) x = (float)(nx() % 2000) / 1000.0f - 1.0f;
    for (auto& x : v) x = (float)(nx() % 2000) / 1000.0f - 1.0f;
    std::vector<int32_t> pos(SEQ);
    for (int i = 0; i < SEQ; i++) pos[i] = i;
    float *d_q, *d_k, *d_v, *o1, *o2;
    int32_t *d_p, *d_p1;
    unsigned char *kc, *vo, *vn;
    CK(hipMalloc(&d_q, q.size() * 4)); CK(hipMalloc(&d_k, k.size() * 4));
    CK(hipMalloc(&d_v, v.size() * 4)); CK(hipMalloc(&o1, NH * HD * 4));
    CK(hipMalloc(&o2, NH * HD * 4)); CK(hipMalloc(&d_p, SEQ * 4));
    CK(hipMalloc(&d_p1, 4)); CK(hipMalloc(&kc, CAP * ROW));
    CK(hipMalloc(&vo, CAP * ROW)); CK(hipMalloc(&vn, CAP * ROW));
    CK(hipMemcpy(d_q, q.data(), q.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_k, k.data(), k.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_v, v.data(), v.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_p, pos.data(), SEQ * 4, hipMemcpyHostToDevice));
    int32_t plast = SEQ - 1;
    CK(hipMemcpy(d_p1, &plast, 4, hipMemcpyHostToDevice));
    { dim3 g(NKV, SEQ, 1), b(32, 1, 1);
      hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_batched, g, b, 0, 0, kc, d_k, d_p, NKV, HD, SEQ);
      hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_batched, g, b, 0, 0, vo, d_v, d_p, NKV, HD, SEQ);
      hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_v_batched, g, b, 0, 0, vn, d_v, d_p, NKV, HD, SEQ);
      CK(hipDeviceSynchronize()); }
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const int IT = 300;
    dim3 g(NH, 1, 1), b(256, 1, 1);
    size_t lds = (2048 + 256 + 256) * 4;
    float ms_old, ms_new;
    CK(hipEventRecord(e0));
    for (int i = 0; i < IT; i++)
        hipLaunchKernelGGL(attention_fp8_e4m3_kv_old, g, b, lds, 0, d_q, kc, vo, o1, d_p1, NH, NKV, HD, CAP, sc);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms_old, e0, e1));
    CK(hipEventRecord(e0));
    for (int i = 0; i < IT; i++)
        hipLaunchKernelGGL(attention_fp8_e4m3_kv, g, b, lds, 0, d_q, kc, vn, o2, d_p1, NH, NKV, HD, CAP, sc);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms_new, e0, e1));
    printf("old %.3f ms/launch | new %.3f ms/launch | delta %+.2f%%\n",
        ms_old / IT, ms_new / IT, (ms_new - ms_old) / ms_old * 100.0);
    return 0;
}
