// Load-time weight requantization into the MQ G256 V2 formats (experiments:
// `Gpu::requant_g256`). Sources expand to F32 (MQ sources stay in their
// rotated basis, the others are rotated afterwards with mq_rotate_x), then
// pack per 128-wide half exactly like hipfire-quantize's MQ V2 encoders:
// step = (hi - lo) / (2^bits - 1) and zero = lo round-trip through f16,
// q = floor((v - zero) / step + 0.5) clamped; a degenerate half stores
// scale 0 and q 0. Payloads are little-endian bitstreams of `bits`-wide
// codes after the 8-byte header, half h at byte 8 + h * 16 * bits.
#include <hip/hip_runtime.h>

static __device__ __forceinline__ float f16_at(const unsigned char* p) {
    return (float)*reinterpret_cast<const _Float16*>(p);
}

extern "C" __global__ void requant_bf16_to_f32(
    const unsigned short* __restrict__ w, float* __restrict__ out, long long n, int) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((unsigned int)w[i] << 16);
}

extern "C" __global__ void requant_q8_0_to_f32(
    const unsigned char* __restrict__ w, float* __restrict__ out, long long n, int) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned char* b = w + (i >> 5) * 34;
    out[i] = f16_at(b) * (float)(signed char)b[2 + (i & 31)];
}

extern "C" __global__ void requant_mqg256v2_to_f32(
    const unsigned char* __restrict__ w, float* __restrict__ out, long long n, int bits) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int j = (int)(i & 255);
    const unsigned char* g = w + (i >> 8) * (8 + 32 * bits);
    const int bit = j * bits;
    const unsigned char* b = g + 8 + (bit >> 3);
    const unsigned int word = b[0] | ((bit & 7) + bits > 8 ? (unsigned int)b[1] << 8 : 0u);
    const int q = (word >> (bit & 7)) & ((1 << bits) - 1);
    const int h = (j >> 7) * 4;
    out[i] = (float)q * f16_at(g + h) + f16_at(g + h + 2);
}

// One thread per 128-wide half of a rotated 256 group.
extern "C" __global__ void requant_pack_mqg256v2(
    const float* __restrict__ x, unsigned char* __restrict__ out, long long halves, int bits) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= halves) return;
    const int h = (int)(t & 1);
    const int levels = (1 << bits) - 1;
    const float* v = x + t * 128;
    unsigned char* g = out + (t >> 1) * (8 + 32 * bits);
    float lo = v[0], hi = v[0];
    for (int i = 1; i < 128; ++i) {
        lo = fminf(lo, v[i]);
        hi = fmaxf(hi, v[i]);
    }
    const float step = hi > lo ? (hi - lo) / (float)levels : 0.0f;
    const _Float16 s16 = hi == lo ? (_Float16)0.0f : (_Float16)step;
    const _Float16 z16 = (_Float16)lo;
    const float st = (float)s16, z = (float)z16;
    *reinterpret_cast<_Float16*>(g + h * 4) = s16;
    *reinterpret_cast<_Float16*>(g + h * 4 + 2) = z16;
    const bool degenerate = hi == lo || step == 0.0f || st == 0.0f;
    const float inv = degenerate ? 0.0f : 1.0f / st;
    unsigned char* p = g + 8 + h * 16 * bits;
    unsigned int acc = 0;
    int filled = 0;
    for (int i = 0; i < 128; ++i) {
        const unsigned int q = degenerate
            ? 0u
            : (unsigned int)fminf(fmaxf(floorf((v[i] - z) * inv + 0.5f), 0.0f), (float)levels);
        acc |= q << filled;
        filled += bits;
        while (filled >= 8) {
            *p++ = (unsigned char)acc;
            acc >>= 8;
            filled -= 8;
        }
    }
}
