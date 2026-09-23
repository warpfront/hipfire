
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

// Batched argmax over F32 rows. Grid.x is the batch dim (one block per row);
// each block reduces `n` values via a per-block shared-memory tournament.
//
// Used in the DFlash spec-decode verify path to fold B argmaxes into a
// single launch that lands just B × 4 bytes on the host — avoiding the
// ~15 MB-per-verify D2H of full logits on the 4B Q8 lm_head.
extern "C" __global__ void argmax_f32_batched(
    const float* __restrict__ data,   // [batch, n] row-major
    int* __restrict__ result,          // [batch]
    int n
) {
    extern __shared__ float s[];
    int* si = (int*)(s + blockDim.x);
    const int batch_idx = blockIdx.x;
    const float* row = data + (size_t)batch_idx * n;

    float lmax = -1e30f;
    int lidx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        if (v > lmax) { lmax = v; lidx = i; }
    }
    s[threadIdx.x] = lmax;
    si[threadIdx.x] = lidx;
    __syncthreads();

    for (int sz = blockDim.x / 2; sz > 0; sz >>= 1) {
        if (threadIdx.x < sz && s[threadIdx.x + sz] > s[threadIdx.x]) {
            s[threadIdx.x] = s[threadIdx.x + sz];
            si[threadIdx.x] = si[threadIdx.x + sz];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) result[batch_idx] = si[0];
}
