#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#define HIPFIRE_KV_FP8_E4M3 1
#include "../../../kernels/src/kv_cache_write_q8_0.hip"
#include "../../../kernels/src/kv_cache_write_q8_0_batched.hip"
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP fail %d: %s\n", __LINE__, hipGetErrorString(e)); return 1; } } while (0)
static uint64_t s = 0x12345678u;
static uint64_t nx() {
    s += 0x9e3779b97f4a7c15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
// host-exact scale rule mirror (f32 max is order-independent; f32->f16 RNE)
static uint16_t host_scale(float amax) {
    _Float16 s16 = (amax == 0.0f) ? (_Float16)1.0f : (_Float16)(amax / 448.0f);
    if (amax > 0.0f && s16 == (_Float16)0.0f)
        s16 = __builtin_bit_cast(_Float16, (unsigned short)0x0001);
    float sf = (float)s16;
    if (amax > 0.0f && 448.0f * sf < amax) {
        unsigned short bits = __builtin_bit_cast(unsigned short, s16);
        bits = (bits >= 0x7bff) ? (unsigned short)0x7bff : (unsigned short)(bits + 1u);
        s16 = __builtin_bit_cast(_Float16, bits);
    }
    return __builtin_bit_cast(unsigned short, s16);
}
int main() {
    CK(hipSetDevice(0));
    const int NKV = 4, HD = 256, B = 16;
    const size_t ROW = (size_t)NKV * (HD + 2);
    const size_t CODES = (size_t)NKV * HD;
    const int CAP = 16;
    unsigned char *vo, *vn;
    CK(hipMalloc(&vo, CAP * ROW));
    CK(hipMalloc(&vn, CAP * ROW));
    CK(hipMemset(vo, 0, CAP * ROW));
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
    float* dv;
    int32_t* dp;
    CK(hipMalloc(&dv, hv.size() * 4));
    CK(hipMalloc(&dp, B * 4));
    CK(hipMemcpy(dv, hv.data(), hv.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(dp, pos.data(), B * 4, hipMemcpyHostToDevice));
    {
        dim3 grid(NKV, B, 1), block(32, 1, 1);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_batched, grid, block, 0, 0,
            vo, dv, dp, NKV, HD, B);
        hipLaunchKernelGGL(kv_cache_write_fp8_e4m3_v_batched, grid, block, 0, 0,
            vn, dv, dp, NKV, HD, B);
        CK(hipDeviceSynchronize());
    }
    std::vector<unsigned char> ho(CAP * ROW), hn(CAP * ROW);
    CK(hipMemcpy(ho.data(), vo, CAP * ROW, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hn.data(), vn, CAP * ROW, hipMemcpyDeviceToHost));
    // host truth for every (t,h) scale
    for (int t = 0; t < B; t++)
        for (int h = 0; h < NKV; h++) {
            float amax = 0;
            for (int d = 0; d < HD; d++) {
                float v = hv[(size_t)t * NKV * HD + h * HD + d];
                if (fabsf(v) > amax) amax = fabsf(v);
            }
            uint16_t exp = host_scale(amax);
            uint16_t got_o = ho[(size_t)t * ROW + CODES + h * 2]
                | ((uint16_t)ho[(size_t)t * ROW + CODES + h * 2 + 1] << 8);
            uint16_t got_n = hn[(size_t)t * ROW + CODES + h * 2]
                | ((uint16_t)hn[(size_t)t * ROW + CODES + h * 2 + 1] << 8);
            if (exp != got_o || exp != got_n)
                printf("t=%d h=%d amax=%.6f exp=%04x old=%04x new=%04x %s%s\n",
                    t, h, amax, exp, got_o, got_n,
                    exp == got_o ? "" : " OLD-WRONG",
                    exp == got_n ? "" : " NEW-WRONG");
        }
    printf("scale-truth done\n");
    return 0;
}
