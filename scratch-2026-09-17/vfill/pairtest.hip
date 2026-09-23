#include <hip/hip_runtime.h>
#include <cstdio>
typedef float f32x2 __attribute__((ext_vector_type(2)));
__global__ void k(unsigned int w, float* out) {
    f32x2 p0 = __builtin_amdgcn_cvt_pk_f32_fp8(w, 0);
    f32x2 p1 = __builtin_amdgcn_cvt_pk_f32_fp8(w, 1);
    out[0] = p0[0]; out[1] = p0[1]; out[2] = p1[0]; out[3] = p1[1];
}
int main() {
    hipSetDevice(0);
    float* d; hipMalloc(&d, 4 * 4);
    // bytes: b0=0x3C (1.0), b1=0xBC (-1.0), b2=0x44 (3.0), b3=0xC4 (-3.0)
    // E4M3: 0x3C = 0 0111 100 = e7 m4: 2^0*(1+4/8)=1.5? compute: e=7,m=4 -> 1.5. use known instead:
    // 0x38: e=7,m=0 -> 1.0 ; 0xB8: -1.0 ; 0x40: e=8,m=0 -> 2.0 ; 0xC0: -2.0
    unsigned int w = 0xC040B838u; // b0=0x38(1.0) b1=0xB8(-1.0) b2=0x40(2.0) b3=0xC0(-2.0)
    hipLaunchKernelGGL(k, dim3(1), dim3(1), 0, 0, w, d);
    hipDeviceSynchronize();
    float h[4]; hipMemcpy(h, d, 16, hipMemcpyDeviceToHost);
    printf("sel0: %f %f | sel1: %f %f\n", h[0], h[1], h[2], h[3]);
    printf("expect sel0: 1.0 -1.0 | sel1: 2.0 -2.0\n");
    return 0;
}
