#include <hip/hip_runtime.h>
extern "C" __global__ void nt_store_probe(float* p, float x) {
    __builtin_nontemporal_store(x, p + threadIdx.x);
}
extern "C" __global__ void normal_store_probe(float* p, float x) {
    p[threadIdx.x] = x;
}
