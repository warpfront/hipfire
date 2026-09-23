// Probe v4: (dword, constant-select) form + host include.
#include <hip/hip_runtime.h>
typedef float f32x2 __attribute__((ext_vector_type(2)));
__global__ void probe_pk_f32_fp8(const unsigned int* in, float* out) {
    unsigned int w = in[0];
    f32x2 v = __builtin_amdgcn_cvt_pk_f32_fp8(w, 0);
    out[0] = v[0];
    out[1] = v[1];
    f32x2 v2 = __builtin_amdgcn_cvt_pk_f32_fp8(w, 1);
    out[2] = v2[0];
    out[3] = v2[1];
}
