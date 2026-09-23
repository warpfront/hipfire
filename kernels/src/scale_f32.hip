#include <hip/hip_runtime.h>
extern "C" __global__ void scale_f32(float* __restrict__ x, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}
