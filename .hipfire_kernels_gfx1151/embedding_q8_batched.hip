
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

// Batched Q8_0 embedding lookup. Dequantizes N rows of the Q8 embedding
// table into `output[n, dim]` in a single launch, reading token ids from
// a device buffer so the launch is hipGraph-captureable.
extern "C" __global__ void embedding_q8_batched(
    const unsigned char* __restrict__ table,    // [vocab × bytes_per_row]
    float* __restrict__ output,                 // [n × dim] row-major
    const int* __restrict__ token_ids,          // [n]
    int dim
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int blocks_per_row = dim / 32;
    const int bytes_per_row = blocks_per_row * 34;
    const int token_id = token_ids[batch_idx];
    const unsigned char* row = table + (size_t)token_id * bytes_per_row;
    float* out_row = output + (size_t)batch_idx * dim;

    for (int elem = tid; elem < dim; elem += blockDim.x) {
        int block_idx = elem / 32;
        int within = elem % 32;
        const unsigned char* block = row + block_idx * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + within];
        out_row[elem] = d * (float)qval;
    }
}
