// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Segment twins for cross-request batched speculative verify.
//
// Appended to a singleton kernel's own source whose entry was rewritten into
// an inlined device function (`*_seg_body`, see rdna-compute
// `verify_twins.rs`). One request's verify-row segment per blockIdx.z: the
// wrapper loads that segment's pointers and row count from a table and runs
// exactly the singleton launch's block over them (the singleton's grid x/y
// and block), so every segment is the singleton launch by construction.
// Blocks past a segment's row count (ragged segments) exit uniformly.

#ifdef HIPFIRE_SEGS_ATTN_FP8
// One request's `attention_fp8_e4m3_kv_batched` launch (Grid [n_heads,
// n_rows]); `n_rows` occupies a full u64 slot to keep the table u64-only.
struct HfAttnFp8Seg {
    const float* q;
    const unsigned char* k_cache;
    const unsigned char* v_cache;
    float* out;
    const int* positions;
    unsigned long long n_rows;
};

extern "C" __global__ void attention_fp8_e4m3_kv_batched_segs(
    const HfAttnFp8Seg* __restrict__ segs,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn)
{
    const HfAttnFp8Seg s = segs[blockIdx.z];
    if ((unsigned long long)blockIdx.y >= s.n_rows) return;
    attention_fp8_e4m3_kv_batched_seg_body(
        s.q, s.k_cache, s.v_cache, s.out, s.positions, nullptr,
        n_heads, n_kv_heads, head_dim, max_seq, scale_attn, 0, 0, nullptr, nullptr);
}
#endif
