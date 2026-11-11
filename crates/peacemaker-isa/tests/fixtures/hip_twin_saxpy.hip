// Fixture source for tests/hip_twin.rs. Built once with:
//   /opt/rocm/bin/hipcc --genco --offload-arch=gfx1100 -O3 --no-offload-compress -fuse-cuid=none \
//     hip_twin_saxpy.hip -o hip_twin_saxpy_gfx1100.co
#include <hip/hip_runtime.h>
extern "C" __global__ void twin_saxpy(float* y, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
extern "C" __global__ void twin_saxpy2(float* y, const float* x, float a, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] - y[i];
}
