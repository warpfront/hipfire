#include <hip/hip_runtime.h>

// Native ds_swizzle_b32 Hadamard butterfly: XOR swap + add/sub.
// Replaces __shfl_xor (compiles to ds_bpermute, needs VGPR for lane index).
// ds_swizzle encodes XOR pattern as immediate — no VGPR, lower latency.
#define HADAMARD_BFLY(v, pattern, stride, tid) do {                     \
    float _p = __int_as_float(                                          \
        __builtin_amdgcn_ds_swizzle(__float_as_int(v), (pattern)));     \
    if ((tid) & (stride)) { (v) = _p - (v); }                          \
    else                  { (v) = (v) + _p; }                           \
} while(0)

// Lloyd-Max optimal centroids for N(0, 1/128) after unit-norm + FWHT(1/sqrt(128))
__constant__ float TURBO_C2[4] = {-0.133466f, -0.040022f, 0.040022f, 0.133466f};
__constant__ float TURBO_C3[8] = {-0.190685f, -0.117832f, -0.065717f, -0.021460f, 0.021460f, 0.065717f, 0.117832f, 0.190685f};
__constant__ float TURBO_C4[16] = {
    -0.241565f, -0.182875f, -0.143012f, -0.111016f, -0.083262f, -0.057983f, -0.034295f, -0.011225f,
     0.011225f,  0.034295f,  0.057983f,  0.083262f,  0.111016f,  0.143012f,  0.182875f,  0.241565f
};

// Lloyd-Max optimal centroids for N(0, 1/256) — all values = 128-dim × 1/sqrt(2)
__constant__ float TURBO_C2_256[4] = {-0.094376f, -0.028300f, 0.028300f, 0.094376f};
__constant__ float TURBO_C3_256[8] = {-0.134860f, -0.083320f, -0.046469f, -0.015176f, 0.015176f, 0.046469f, 0.083320f, 0.134860f};
__constant__ float TURBO_C4_256[16] = {
    -0.170807f, -0.129321f, -0.101134f, -0.078505f, -0.058869f, -0.041003f, -0.024249f, -0.007938f,
     0.007938f,  0.024249f,  0.041003f,  0.058869f,  0.078505f,  0.101134f,  0.129321f,  0.170807f
};

// In-place FWHT on 128 elements in registers.
// signs1/signs2 are ±1.0f arrays in global memory (uploaded once).
__device__ void fwht_forward_128(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    // Step 1: apply signs1
    for (int i = 0; i < 128; i++) x[i] *= signs1[i];

    // Step 2: Walsh-Hadamard butterfly (7 passes for n=128)
    for (int stride = 1; stride < 128; stride <<= 1) {
        for (int i = 0; i < 128; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }

    // Step 3: scale by 1/sqrt(128)
    const float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;

    // Step 4: apply signs2
    for (int i = 0; i < 128; i++) x[i] *= signs2[i];
}

// Inverse FWHT: signs2 -> butterfly -> scale -> signs1 (reverse order)
__device__ void fwht_inverse_128(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 128; i++) x[i] *= signs2[i];
    for (int stride = 1; stride < 128; stride <<= 1) {
        for (int i = 0; i < 128; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128 * signs1[i];
}


// Register-only FWHT via __shfl_xor. Zero shared memory. Zero barriers.
// Each thread owns 4 of 128 elements in registers (a,b,c,d).
// signs1/signs2 are applied from constant memory.
// After this function, (a,b,c,d) are in the FWHT-rotated space.
__device__ void fwht_shfl_forward(float& a, float& b, float& c, float& d,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    int d0 = tid * 4;
    // Apply signs1
    a *= signs1[d0]; b *= signs1[d0+1]; c *= signs1[d0+2]; d *= signs1[d0+3];

    // Local butterfly: stride 1 (pairs 0↔1, 2↔3)
    float t;
    t = a; a = a + b; b = t - b;
    t = c; c = c + d; d = t - d;

    // Local butterfly: stride 2 (pairs 0↔2, 1↔3)
    t = a; a = a + c; c = t - c;
    t = b; b = b + d; d = t - d;

    // Wave-level butterfly via ds_swizzle_b32 (native XOR swap, no VGPR for lane index)
    HADAMARD_BFLY(a, 0x041F, 1, tid); HADAMARD_BFLY(b, 0x041F, 1, tid);
    HADAMARD_BFLY(c, 0x041F, 1, tid); HADAMARD_BFLY(d, 0x041F, 1, tid);
    HADAMARD_BFLY(a, 0x081F, 2, tid); HADAMARD_BFLY(b, 0x081F, 2, tid);
    HADAMARD_BFLY(c, 0x081F, 2, tid); HADAMARD_BFLY(d, 0x081F, 2, tid);
    HADAMARD_BFLY(a, 0x101F, 4, tid); HADAMARD_BFLY(b, 0x101F, 4, tid);
    HADAMARD_BFLY(c, 0x101F, 4, tid); HADAMARD_BFLY(d, 0x101F, 4, tid);
    HADAMARD_BFLY(a, 0x201F, 8, tid); HADAMARD_BFLY(b, 0x201F, 8, tid);
    HADAMARD_BFLY(c, 0x201F, 8, tid); HADAMARD_BFLY(d, 0x201F, 8, tid);
    HADAMARD_BFLY(a, 0x401F, 16, tid); HADAMARD_BFLY(b, 0x401F, 16, tid);
    HADAMARD_BFLY(c, 0x401F, 16, tid); HADAMARD_BFLY(d, 0x401F, 16, tid);

    // Scale by 1/sqrt(128) and apply signs2
    const float s = 0.08838834764831845f;
    a *= s * signs2[d0]; b *= s * signs2[d0+1]; c *= s * signs2[d0+2]; d *= s * signs2[d0+3];
}

// Inverse: signs2, butterfly, scale, signs1 (reverse order)
__device__ void fwht_shfl_inverse(float& a, float& b, float& c, float& d,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    int d0 = tid * 4;
    a *= signs2[d0]; b *= signs2[d0+1]; c *= signs2[d0+2]; d *= signs2[d0+3];

    HADAMARD_BFLY(a, 0x041F, 1, tid); HADAMARD_BFLY(b, 0x041F, 1, tid);
    HADAMARD_BFLY(c, 0x041F, 1, tid); HADAMARD_BFLY(d, 0x041F, 1, tid);
    HADAMARD_BFLY(a, 0x081F, 2, tid); HADAMARD_BFLY(b, 0x081F, 2, tid);
    HADAMARD_BFLY(c, 0x081F, 2, tid); HADAMARD_BFLY(d, 0x081F, 2, tid);
    HADAMARD_BFLY(a, 0x101F, 4, tid); HADAMARD_BFLY(b, 0x101F, 4, tid);
    HADAMARD_BFLY(c, 0x101F, 4, tid); HADAMARD_BFLY(d, 0x101F, 4, tid);
    HADAMARD_BFLY(a, 0x201F, 8, tid); HADAMARD_BFLY(b, 0x201F, 8, tid);
    HADAMARD_BFLY(c, 0x201F, 8, tid); HADAMARD_BFLY(d, 0x201F, 8, tid);
    HADAMARD_BFLY(a, 0x401F, 16, tid); HADAMARD_BFLY(b, 0x401F, 16, tid);
    HADAMARD_BFLY(c, 0x401F, 16, tid); HADAMARD_BFLY(d, 0x401F, 16, tid);

    float t;
    t = a; a = a + c; c = t - c;
    t = b; b = b + d; d = t - d;
    t = a; a = a + b; b = t - b;
    t = c; c = c + d; d = t - d;

    const float s = 0.08838834764831845f;
    a *= s * signs1[d0]; b *= s * signs1[d0+1]; c *= s * signs1[d0+2]; d *= s * signs1[d0+3];
}

// ============================================================================
// 256-element FWHT via __shfl_xor. Zero shared memory. Zero barriers.
// Each thread owns 8 of 256 elements in registers (v0..v7).
// 32 threads × 8 elements = 256.
// log2(256) = 8 butterfly passes:
//   Passes 1-3: register-local (strides 1, 2, 4 in element space)
//   Passes 4-8: warp shuffle    (strides 8,16,32,64,128 in element space
//                                = strides 1,2,4,8,16 in thread space)
//
// VGPR budget (RDNA, 32-lane wavefront):
//   v0-v7:  8 VGPRs  (the 8 data elements)
//   t:      1 VGPR   (butterfly temp, reused across all local passes)
//   d0:     1 VGPR   (base index for sign table loads, computed once)
//   s:      1 VGPR   (scale constant 1/sqrt(256), fused with signs2)
//   p0-p7:  0 VGPRs  (shuffle results alias v0-v7 via compiler; see note)
//   tid:    1 VGPR   (passed in, likely already live in caller)
//   signs:  0 VGPRs  (loaded directly into multiply operand from global)
//   -------
//   Total: 12 VGPRs kernel-private + 8 data = 20 VGPRs worst case
//
// Note on shuffle passes: each __shfl_xor result (p0..p7) is consumed
// immediately in the add/sub that overwrites v0..v7, so the compiler
// can alias them. At most 1 extra VGPR for the shuffle temporary if
// the compiler doesn't overlap. In practice expect 18-19 VGPRs on RDNA.
// ============================================================================

__device__ void fwht_shfl_forward_256(
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    const int d0 = tid * 8;

    // Apply signs1 (randomize input for incoherence)
    v0 *= signs1[d0];   v1 *= signs1[d0+1]; v2 *= signs1[d0+2]; v3 *= signs1[d0+3];
    v4 *= signs1[d0+4]; v5 *= signs1[d0+5]; v6 *= signs1[d0+6]; v7 *= signs1[d0+7];

    // --- Pass 1: stride 1 in element space ---
    // Butterfly pairs: (0,1), (2,3), (4,5), (6,7)
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    // --- Pass 2: stride 2 in element space ---
    // Butterfly pairs: (0,2), (1,3), (4,6), (5,7)
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    // --- Pass 3: stride 4 in element space ---
    // Butterfly pairs: (0,4), (1,5), (2,6), (3,7)
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // --- Passes 4-8: ds_swizzle_b32 (native XOR swap, no VGPR for lane index) ---
    #define HBFLY8(pat, str) \
        HADAMARD_BFLY(v0,(pat),(str),tid); HADAMARD_BFLY(v1,(pat),(str),tid); \
        HADAMARD_BFLY(v2,(pat),(str),tid); HADAMARD_BFLY(v3,(pat),(str),tid); \
        HADAMARD_BFLY(v4,(pat),(str),tid); HADAMARD_BFLY(v5,(pat),(str),tid); \
        HADAMARD_BFLY(v6,(pat),(str),tid); HADAMARD_BFLY(v7,(pat),(str),tid)
    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8

    // Scale by 1/sqrt(256) = 1/16 = 0.0625 and apply signs2
    const float s = 0.0625f;
    v0 *= s * signs2[d0];   v1 *= s * signs2[d0+1]; v2 *= s * signs2[d0+2]; v3 *= s * signs2[d0+3];
    v4 *= s * signs2[d0+4]; v5 *= s * signs2[d0+5]; v6 *= s * signs2[d0+6]; v7 *= s * signs2[d0+7];
}

// Inverse FWHT-256: signs2 -> shuffle passes -> local passes (reversed) -> scale -> signs1
__device__ void fwht_shfl_inverse_256(
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int tid)
{
    const int d0 = tid * 8;

    // Apply signs2 (undo output randomization)
    v0 *= signs2[d0];   v1 *= signs2[d0+1]; v2 *= signs2[d0+2]; v3 *= signs2[d0+3];
    v4 *= signs2[d0+4]; v5 *= signs2[d0+5]; v6 *= signs2[d0+6]; v7 *= signs2[d0+7];

    // --- Passes 4-8: ds_swizzle_b32 ---
    #define HBFLY8(pat, str) \
        HADAMARD_BFLY(v0,(pat),(str),tid); HADAMARD_BFLY(v1,(pat),(str),tid); \
        HADAMARD_BFLY(v2,(pat),(str),tid); HADAMARD_BFLY(v3,(pat),(str),tid); \
        HADAMARD_BFLY(v4,(pat),(str),tid); HADAMARD_BFLY(v5,(pat),(str),tid); \
        HADAMARD_BFLY(v6,(pat),(str),tid); HADAMARD_BFLY(v7,(pat),(str),tid)
    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8

    // --- Pass 3 (reverse): stride 4 --- pairs (0,4), (1,5), (2,6), (3,7)
    float t;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // --- Pass 2 (reverse): stride 2 --- pairs (0,2), (1,3), (4,6), (5,7)
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    // --- Pass 1 (reverse): stride 1 --- pairs (0,1), (2,3), (4,5), (6,7)
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    // Scale by 1/sqrt(256) and apply signs1
    const float s = 0.0625f;
    v0 *= s * signs1[d0];   v1 *= s * signs1[d0+1]; v2 *= s * signs1[d0+2]; v3 *= s * signs1[d0+3];
    v4 *= s * signs1[d0+4]; v5 *= s * signs1[d0+5]; v6 *= s * signs1[d0+6]; v7 *= s * signs1[d0+7];
}

// ============================================================================
// Scalar reference FWHT-256 (for validation / non-turbo paths)
// ============================================================================
__device__ void fwht_forward_256(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 256; i++) x[i] *= signs1[i];
    for (int stride = 1; stride < 256; stride <<= 1) {
        for (int i = 0; i < 256; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_256 = 0.0625f;
    for (int i = 0; i < 256; i++) x[i] *= inv_sqrt_256 * signs2[i];
}

__device__ void fwht_inverse_256(float* x,
    const float* __restrict__ signs1, const float* __restrict__ signs2)
{
    for (int i = 0; i < 256; i++) x[i] *= signs2[i];
    for (int stride = 1; stride < 256; stride <<= 1) {
        for (int i = 0; i < 256; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = x[i + j];
                float b = x[i + j + stride];
                x[i + j]          = a + b;
                x[i + j + stride] = a - b;
            }
        }
    }
    const float inv_sqrt_256 = 0.0625f;
    for (int i = 0; i < 256; i++) x[i] *= inv_sqrt_256 * signs1[i];
}


// Branchless 2-bit quantize: returns index 0-3 (thresholds for N(0, 1/128))
__device__ int turbo_quantize_2bit(float x) {
    return (x > -0.086744f) + (x > 0.0f) + (x > 0.086744f);
}

// Branchless 3-bit quantize: returns index 0-7
__device__ int turbo_quantize_3bit(float x) {
    return (x > -0.154258f) + (x > -0.091775f) + (x > -0.043589f) + (x > 0.0f)
         + (x > 0.043589f) + (x > 0.091775f) + (x > 0.154258f);
}

// Branchless 4-bit quantize: returns index 0-15
__device__ int turbo_quantize_4bit(float x) {
    return (x > -0.212220f) + (x > -0.162944f) + (x > -0.127014f) + (x > -0.097139f)
         + (x > -0.070622f) + (x > -0.046139f) + (x > -0.022760f) + (x > 0.0f)
         + (x > 0.022760f) + (x > 0.046139f) + (x > 0.070622f) + (x > 0.097139f)
         + (x > 0.127014f) + (x > 0.162944f) + (x > 0.212220f);
}

// 256-dim quantize functions — thresholds = 128-dim × 1/sqrt(2)
__device__ int turbo_quantize_2bit_256(float x) {
    return (x > -0.061352f) + (x > 0.0f) + (x > 0.061352f);
}
__device__ int turbo_quantize_3bit_256(float x) {
    return (x > -0.109068f) + (x > -0.064903f) + (x > -0.030827f) + (x > 0.0f)
         + (x > 0.030827f) + (x > 0.064903f) + (x > 0.109068f);
}
__device__ int turbo_quantize_4bit_256(float x) {
    return (x > -0.150086f) + (x > -0.115239f) + (x > -0.089815f) + (x > -0.068691f)
         + (x > -0.049935f) + (x > -0.032626f) + (x > -0.016096f) + (x > 0.0f)
         + (x > 0.016096f) + (x > 0.032626f) + (x > 0.049935f) + (x > 0.068691f)
         + (x > 0.089815f) + (x > 0.115239f) + (x > 0.150086f);
}

// Sign flip array for cheap decorrelation (seed=42, ±1.0)
__constant__ float TURBO_SIGNS1[128] = {
  1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f,
  1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,
 -1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f,
  1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,
  1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,
  1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f,
  1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f
};

// Block-diagonal 2x2 Givens rotation primitives for givens4 KV cache.
//
// Thread tid owns 4 consecutive dims [tid*4..tid*4+3].
// Two independent Givens blocks per thread:
//   Block A: (a, b) rotated by angle[tid*2]
//   Block B: (c, d) rotated by angle[tid*2+1]
//
// cos_theta, sin_theta: [head_dim/2] float arrays (64 entries for hd=128).
// Purely register-local — no wave shuffles, no LDS, no cross-thread deps.
#pragma once

__device__ __forceinline__ void givens_forward(
    float& a, float& b, float& c, float& d,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta, int tid)
{
    int b0 = tid * 2;
    float c0 = cos_theta[b0],   s0 = sin_theta[b0];
    float c1 = cos_theta[b0+1], s1 = sin_theta[b0+1];
    // Block A: rotate (a, b)
    float a2 = a * c0 - b * s0;
    float b2 = a * s0 + b * c0;
    a = a2; b = b2;
    // Block B: rotate (c, d)
    float c2 = c * c1 - d * s1;
    float d2 = c * s1 + d * c1;
    c = c2; d = d2;
}

__device__ __forceinline__ void givens_inverse(
    float& a, float& b, float& c, float& d,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta, int tid)
{
    int b0 = tid * 2;
    float c0 = cos_theta[b0],   s0 = sin_theta[b0];
    float c1 = cos_theta[b0+1], s1 = sin_theta[b0+1];
    // Inverse = transpose: negate sin
    float a2 =  a * c0 + b * s0;
    float b2 = -a * s0 + b * c0;
    a = a2; b = b2;
    float c2 =  c * c1 + d * s1;
    float d2 = -c * s1 + d * c1;
    c = c2; d = d2;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// K-only write for asym3: Givens rotation + 3-bit quantize (Lloyd-Max N(0, 1/256)).
// V is written separately via kv_cache_write_q8_0.
//
// Per RotorQuant: 3-bit gives the best quality/compression tradeoff at
// rotated K — 8 centroids cover the N(0, 1/d) distribution well without
// wasting codebook on empty tails (as 4-bit does). Paper reports 3-bit
// K planar + FP16 V ≈ 0% PPL loss.
//
// Head layout (head_dim=256): 32 threads × 8 values = 256 dims covered
// in a single pass. 8 × 3-bit = 24 bits = 3 bytes per thread.
//   [4B cnorm][32 threads × 3 bytes packed indices] = 100 B/head.
//
// Givens: each thread handles 4 2D blocks (indices tid*4 .. tid*4+3).
// cos_theta/sin_theta size = head_dim/2 = 128 entries.
//
// Grid: [n_kv_heads]. Block: [32]. LDS: (head_dim + 32) * 4 bytes.
#include <hip/hip_runtime.h>



__device__ __forceinline__ void givens_rot_inplace(
    float& a, float& b,
    float c, float s)
{
    float a2 = a * c - b * s;
    float b2 = a * s + b * c;
    a = a2; b = b2;
}

extern "C" __global__ void kv_cache_write_asym_k_givens3(
    unsigned char* __restrict__ k_dst,
    const float* __restrict__ k_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    int n_kv_heads, int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int pos = pos_buf[0];

    // 3-bit packed: 4B cnorm + 32 threads × 3 bytes
    const int bytes_per_head = 4 + (head_dim * 3) / 8;   // 100 for hd=256
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;
    float* scratch = smem + head_dim;

    const float* src = k_src + h * head_dim;
    unsigned char* out = k_dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    for (int d = tid; d < head_dim; d += nthreads) x[d] = src[d];
    __syncthreads();

    // Parallel L2 norm
    float lsq = 0.0f;
    for (int d = tid; d < head_dim; d += nthreads) lsq += x[d] * x[d];
    scratch[tid] = lsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float orig_norm = sqrtf(scratch[0]);
    float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

    for (int d = tid; d < head_dim; d += nthreads) x[d] *= inv_norm;
    __syncthreads();

    // Each thread handles 8 consecutive dims = 4 Givens blocks.
    const int d0 = tid * 8;
    float v0 = x[d0 + 0], v1 = x[d0 + 1];
    float v2 = x[d0 + 2], v3 = x[d0 + 3];
    float v4 = x[d0 + 4], v5 = x[d0 + 5];
    float v6 = x[d0 + 6], v7 = x[d0 + 7];

    const int b0 = tid * 4;
    givens_rot_inplace(v0, v1, cos_theta[b0 + 0], sin_theta[b0 + 0]);
    givens_rot_inplace(v2, v3, cos_theta[b0 + 1], sin_theta[b0 + 1]);
    givens_rot_inplace(v4, v5, cos_theta[b0 + 2], sin_theta[b0 + 2]);
    givens_rot_inplace(v6, v7, cos_theta[b0 + 3], sin_theta[b0 + 3]);

    // Quantize 8 values → 3-bit indices. Lloyd-Max centroids for N(0, 1/256).
    int idx[8];
    idx[0] = turbo_quantize_3bit_256(v0);
    idx[1] = turbo_quantize_3bit_256(v1);
    idx[2] = turbo_quantize_3bit_256(v2);
    idx[3] = turbo_quantize_3bit_256(v3);
    idx[4] = turbo_quantize_3bit_256(v4);
    idx[5] = turbo_quantize_3bit_256(v5);
    idx[6] = turbo_quantize_3bit_256(v6);
    idx[7] = turbo_quantize_3bit_256(v7);

    float local_rsq = 0.0f;
    for (int i = 0; i < 8; i++) local_rsq += TURBO_C3_256[idx[i]] * TURBO_C3_256[idx[i]];

    scratch[tid] = local_rsq;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) { if (tid < s) scratch[tid] += scratch[tid + s]; __syncthreads(); }
    float recon_norm = sqrtf(scratch[0]);
    float cnorm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;
    if (tid == 0) *(float*)out = cnorm;

    // Pack 8 × 3-bit = 24 bits = 3 bytes per thread.
    unsigned int packed = 0;
    for (int i = 0; i < 8; i++) packed |= ((unsigned int)(idx[i] & 7)) << (i * 3);
    unsigned char* base = out + 4 + tid * 3;
    base[0] = (unsigned char)(packed       & 0xFF);
    base[1] = (unsigned char)((packed >> 8)  & 0xFF);
    base[2] = (unsigned char)((packed >> 16) & 0xFF);
}
