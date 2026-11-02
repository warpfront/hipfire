// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Symmetric direct-peer-read allreduce residual add for dense TP=2.
//
// Computes `x[i] = x[i] + (p_first[i] + p_second[i])` where one of the two
// partial pointers is a P2P-mapped pointer on the peer device (plain loads
// work under peer access, no memcpy). Each rank runs this on its own stream
// after waiting on the peer's producer event, so no root, no scratch, and
// no host waits.
//
// Bit-exactness: the parenthesization is load-bearing. The rooted impl folds
// `p0 + p1` (rank order) then adds the residual as `x + sum`. Both ranks
// here MUST evaluate the same `p0 + p1` rank order, so the caller passes the
// lower rank's partial as `p_first` (the same two operands on both ranks;
// only `x` differs). Grid-stride loop keeps every lane's accumulation order
// identical to the scalar tail.

#include <hip/hip_runtime.h>

extern "C" __global__ void add_peer_residual_f32(
    float* __restrict__ x,
    const float* __restrict__ p_first,
    const float* __restrict__ p_second,
    int count
) {
    const int vec = count >> 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = i;
    const int stride = gridDim.x * blockDim.x;
    float4* __restrict__ xv = reinterpret_cast<float4*>(x);
    const float4* __restrict__ fv = reinterpret_cast<const float4*>(p_first);
    const float4* __restrict__ sv = reinterpret_cast<const float4*>(p_second);
    for (; i < vec; i += stride) {
        float4 xi = xv[i];
        float4 fi = fv[i];
        float4 si = sv[i];
        xi.x = xi.x + (fi.x + si.x);
        xi.y = xi.y + (fi.y + si.y);
        xi.z = xi.z + (fi.z + si.z);
        xi.w = xi.w + (fi.w + si.w);
        xv[i] = xi;
    }
    // Tail uses the original thread id: `i` has advanced past `vec` for
    // threads that ran the vector loop, which would skip residues.
    for (int j = (vec << 2) + tid; j < count; j += stride) {
        x[j] = x[j] + (p_first[j] + p_second[j]);
    }
}
