// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Packs the ViT fused QKV activations (F32 [N, 3*hidden], Q at head*head_dim+d,
// K at hidden+..., V at 2*hidden+...) into three F16 buffers [N, num_heads,
// hdp], RNE f32->f16, with d in [head_dim, hdp) written as 0. Feeds
// vit_attention_wmma_f16 (hdp = head_dim rounded up to 16, e.g. 72 -> 80).
//
// Grid = [ceil(N*num_heads*hdp / 256)], block = [256]; one thread per output
// element of each of the three buffers.

#include <hip/hip_runtime.h>

extern "C" __global__ void vit_qkv_pack_f16(
    const float *__restrict__ qkv, _Float16 *__restrict__ qh,
    _Float16 *__restrict__ kh, _Float16 *__restrict__ vh, int N, int hidden,
    int num_heads, int head_dim, int hdp) {
    const long long total = (long long)N * num_heads * hdp;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int d = (int)(idx % hdp);
    const long long r = idx / hdp;
    const int head = (int)(r % num_heads);
    const long long n = r / num_heads;
    float q = 0.f, k = 0.f, v = 0.f;
    if (d < head_dim) {
        const float *row = qkv + n * 3LL * hidden + head * head_dim + d;
        q = row[0];
        k = row[hidden];
        v = row[2 * hidden];
    }
    qh[idx] = (_Float16)q;
    kh[idx] = (_Float16)k;
    vh[idx] = (_Float16)v;
}
