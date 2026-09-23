
extern "C" __attribute__((device)) float __ocml_rsqrt_f32(float);
extern "C" __attribute__((device)) float __ocml_sqrt_f32(float);
extern "C" __attribute__((device)) float __ocml_exp_f32(float);
extern "C" __attribute__((device)) float __ocml_log_f32(float);
extern "C" __attribute__((device)) float __ocml_fabs_f32(float);
extern "C" __attribute__((device)) float __ocml_floor_f32(float);
extern "C" __attribute__((device)) float __ocml_fmax_f32(float, float);
extern "C" __attribute__((device)) float __ocml_fmin_f32(float, float);
extern "C" __attribute__((device)) float __ocml_pow_f32(float, float);
extern "C" __attribute__((device)) float __ocml_cos_f32(float);
extern "C" __attribute__((device)) float __ocml_sin_f32(float);
static inline __attribute__((device)) float rsqrtf(float x) { return __ocml_rsqrt_f32(x); }
static inline __attribute__((device)) float sqrtf(float x) { return __ocml_sqrt_f32(x); }
static inline __attribute__((device)) float expf(float x) { return __ocml_exp_f32(x); }
static inline __attribute__((device)) float logf(float x) { return __ocml_log_f32(x); }
static inline __attribute__((device)) float fabsf(float x) { return __ocml_fabs_f32(x); }
static inline __attribute__((device)) float floorf(float x) { return __ocml_floor_f32(x); }
static inline __attribute__((device)) float fmaxf(float a, float b) { return __ocml_fmax_f32(a, b); }
static inline __attribute__((device)) float fminf(float a, float b) { return __ocml_fmin_f32(a, b); }
static inline __attribute__((device)) float powf(float a, float b) { return __ocml_pow_f32(a, b); }
static inline __attribute__((device)) float cosf(float x) { return __ocml_cos_f32(x); }
static inline __attribute__((device)) float sinf(float x) { return __ocml_sin_f32(x); }
static inline __attribute__((device)) double max(double a, double b) { return a > b ? a : b; }
static inline __attribute__((device)) float max(float a, float b) { return a > b ? a : b; }
static inline __attribute__((device)) int max(int a, int b) { return a > b ? a : b; }
static inline __attribute__((device)) int min(int a, int b) { return a < b ? a : b; }

#include <hip/hip_runtime.h>

extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n, float eps
) {
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[blockIdx.x * n + i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int idx = blockIdx.x * n + i;
        out[idx] = x[idx] * weight[i] * rms;
    }
}
