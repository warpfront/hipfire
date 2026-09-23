#include <hip/hip_runtime.h>
#include <stdint.h>
typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
extern "C" __global__ void w64_builtin_probe(const int* in, int* out) {
    const int a = in[threadIdx.x];
    const int b = in[threadIdx.x + 64];
    const int32x4_t z = {0, 0, 0, 0};
    const int32x4_t c = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12(false, a, true, b, z, false);
    *(int32x4_t*)(out + threadIdx.x * 4) = c;
}
