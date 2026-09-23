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
